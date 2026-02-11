"""
SageMaker Inference Script for Qwen3-Omni-30B-A3B-Thinking
Supports optional LoRA adapter loading via ADAPTER_S3_URI environment variable.

Expected payload:
{
    "job_id": "unique-job-identifier",  # optional, auto-generated if not provided
    "s3_uri": "s3://bucket/path/to/video.mp4",
    "user_prompt": "What is happening in this video?",
    "system_prompt": "You are a helpful assistant.",  # optional
    "max_new_tokens": 512,  # optional, default 512
    "fps": 1,  # optional, default 1
    "max_frames": 64,  # optional, default 64
    "max_pixels": 360000  # optional, default 360000 (e.g. 600x600)
}

Response:
{
    "job_id": "unique-job-identifier",
    "response": "The video shows...",
    "status": "success"
}
"""

import os
import json
import tempfile
import logging

import torch
import boto3

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Globals for model/processor (loaded once)
model = None
processor = None
process_mm_info = None

HF_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_LOCAL_DIR = "/tmp/model"
FFMPEG_PATH = "/usr/local/bin/ffmpeg"


def _download_model_from_s3(s3_uri: str, local_dir: str, max_workers: int = 16):
    """Download all files from an S3 prefix to a local directory in parallel."""
    import concurrent.futures

    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    os.makedirs(local_dir, exist_ok=True)

    s3 = boto3.client("s3")
    path = s3_uri[5:]
    bucket = path.split("/", 1)[0]
    prefix = path.split("/", 1)[1] if "/" in path else ""
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    # List all objects
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel_path = obj["Key"][len(prefix):]
            if rel_path:
                keys.append((obj["Key"], rel_path, obj["Size"]))

    if not keys:
        raise FileNotFoundError(f"No files found at {s3_uri}")

    total_bytes = sum(size for _, _, size in keys)
    logger.info(
        f"Downloading {len(keys)} files ({total_bytes / 1e9:.1f} GB) "
        f"from {s3_uri} -> {local_dir} with {max_workers} workers"
    )

    def _download_one(key, rel_path):
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client = boto3.client("s3")
        s3_client.download_file(bucket, key, local_path)
        return rel_path

    downloaded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_one, key, rel_path): rel_path
            for key, rel_path, _ in keys
        }
        for fut in concurrent.futures.as_completed(futures):
            fut.result()  # raise if failed
            downloaded += 1
            if downloaded % 20 == 0 or downloaded == len(keys):
                logger.info(f"  Downloaded {downloaded}/{len(keys)} files")

    logger.info(f"Model download complete: {local_dir}")
    return local_dir


def _download_adapter_from_s3(adapter_s3_uri: str) -> str:
    """Download LoRA adapter files from S3 to a local directory.
    
    Supports both:
      - S3 prefix (directory): s3://bucket/path/to/adapter/
      - Single file: s3://bucket/path/to/adapter/adapter_model.safetensors
    
    Returns local directory path containing the adapter files.
    """
    if not adapter_s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {adapter_s3_uri}")

    local_adapter_dir = "/tmp/lora_adapter"
    os.makedirs(local_adapter_dir, exist_ok=True)

    s3 = boto3.client("s3")
    
    # Parse bucket and prefix
    path = adapter_s3_uri[5:]
    bucket = path.split("/", 1)[0]
    prefix = path.split("/", 1)[1] if "/" in path else ""
    # Ensure prefix ends with / for directory listing
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    # List and download all objects under the prefix
    paginator = s3.get_paginator("list_objects_v2")
    downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Get relative path from prefix
            rel_path = key[len(prefix):]
            if not rel_path:  # skip the prefix itself
                continue
            local_path = os.path.join(local_adapter_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)
            downloaded += 1
            logger.info(f"Downloaded adapter file: {rel_path} ({obj['Size']} bytes)")

    if downloaded == 0:
        raise FileNotFoundError(
            f"No adapter files found at {adapter_s3_uri}. "
            "Ensure the S3 path points to a directory containing adapter_model.safetensors and adapter_config.json."
        )

    # Verify required files exist
    adapter_files = os.listdir(local_adapter_dir)
    has_config = "adapter_config.json" in adapter_files
    has_weights = any(f.startswith("adapter_model") for f in adapter_files)
    if not has_config:
        raise FileNotFoundError(
            f"Missing adapter_config.json. Found files: {adapter_files}"
        )
    if not has_weights:
        raise FileNotFoundError(
            f"Missing adapter weights (adapter_model.safetensors or adapter_model.bin). "
            f"Found files: {adapter_files}"
        )

    logger.info(f"Adapter downloaded: {downloaded} files -> {local_adapter_dir}")
    return local_adapter_dir

def model_fn(model_dir):
    """Load model - called once when container starts.

    Reads MODEL_S3_URI env var to download base model from S3.
    If ADAPTER_S3_URI env var is set, downloads and merges LoRA adapter.
    """
    global model, processor, process_mm_info

    # Determine model path: try S3 first, fall back to HuggingFace
    model_s3_uri = os.environ.get("MODEL_S3_URI", "").strip()
    model_path = None

    if model_s3_uri:
        try:
            model_path = MODEL_LOCAL_DIR
            if not os.path.exists(os.path.join(model_path, "config.json")):
                logger.info(f"Downloading base model from S3: {model_s3_uri}")
                _download_model_from_s3(model_s3_uri, model_path)
            else:
                logger.info(f"Base model already present at {model_path}, skipping download")
        except Exception as e:
            logger.warning(f"S3 model download failed: {e}. Falling back to HuggingFace.")
            model_path = None

    if model_path is None:
        logger.info(f"Loading base model from HuggingFace: {HF_MODEL_ID}")
        model_path = HF_MODEL_ID

    logger.info(f"Loading base model from {model_path}")

    from transformers import Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info as _process_mm_info

    process_mm_info = _process_mm_info

    # Check for flash attention
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    except ImportError:
        attn_impl = "eager"
        logger.info("Using eager attention")

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    # --- LoRA adapter loading ---
    adapter_s3_uri = os.environ.get("ADAPTER_S3_URI", "").strip()
    if adapter_s3_uri:
        logger.info(f"Loading LoRA adapter from: {adapter_s3_uri}")
        from peft import PeftModel

        local_adapter_dir = _download_adapter_from_s3(adapter_s3_uri)
        model = PeftModel.from_pretrained(model, local_adapter_dir)
        logger.info("Merging adapter weights into base model...")
        model = model.merge_and_unload()
        logger.info("Adapter merged and unloaded successfully")

        # Cleanup adapter files
        import shutil
        shutil.rmtree(local_adapter_dir, ignore_errors=True)
    else:
        logger.info("No ADAPTER_S3_URI set — running base model")

    logger.info("Model loaded successfully!")
    return model
    

def input_fn(request_body, request_content_type):
    """Parse input payload."""
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(payload, model):
    """Run inference."""
    global processor, process_mm_info
    
    local_video = None
    
    # Auto-generate job_id if not provided
    import uuid
    job_id = payload.get("job_id") or str(uuid.uuid4())[:8]
    
    try:
        s3_uri = payload["s3_uri"]
        user_prompt = payload["user_prompt"]
        system_prompt = payload.get("system_prompt")
        max_new_tokens = payload.get("max_new_tokens", 512)
        fps = payload.get("fps", 1)
        max_frames = payload.get("max_frames", 64)
        max_pixels = payload.get("max_pixels", 256*28*28)
        
        logger.info(f"[{job_id}] Starting inference for {s3_uri}")
        logger.info(f"[{job_id}] Settings: fps={fps}, max_frames={max_frames}, max_pixels={max_pixels}")
        
        # Download video from S3
        local_video = download_s3_video(s3_uri)
        logger.info(f"[{job_id}] Downloaded video: {os.path.getsize(local_video)} bytes")
        
        # Build conversation with video settings
        conversation = []
        if system_prompt:
            conversation.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": local_video,
                    "fps": fps,
                    "max_frames": max_frames,
                    "max_pixels": max_pixels
                },
                {"type": "text", "text": user_prompt}
            ]
        })
        
        # Process inputs
        processor.image_processor.max_pixels = max_pixels
        USE_AUDIO_IN_VIDEO = True
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        
        # Convert dtypes and move to device
        device = next(model.parameters()).device
        dtype = torch.bfloat16
        
        processed_inputs = {}
        for k, v in inputs.items():
            if hasattr(v, 'to'):
                if v.dtype in [torch.float32, torch.float64]:
                    processed_inputs[k] = v.to(device=device, dtype=dtype)
                else:
                    processed_inputs[k] = v.to(device=device)
            else:
                processed_inputs[k] = v
        
        # Generate
        with torch.no_grad():
            text_ids = model.generate(
                **processed_inputs,
                max_new_tokens=max_new_tokens,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        
        response = processor.batch_decode(
            text_ids[:, processed_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        logger.info(f"[{job_id}] Inference complete")
        return {"job_id": job_id, "response": response, "status": "success"}
    
    except Exception as e:
        import traceback
        logger.error(f"[{job_id}] Inference error: {str(e)}")
        logger.error(traceback.format_exc())
        return {"job_id": job_id, "error": str(e), "status": "error"}
    
    finally:
        # Cleanup temp file
        if local_video and os.path.exists(local_video):
            os.unlink(local_video)


def output_fn(prediction, response_content_type):
    """Format output."""
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")


def download_s3_video(s3_uri: str) -> str:
    """Download video from S3 URI to temp file."""
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    
    ext = os.path.splitext(key)[1] or ".mp4"
    
    s3 = boto3.client("s3")
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    s3.download_file(bucket, key, tmp.name)
    
    return tmp.name
