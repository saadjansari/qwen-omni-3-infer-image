"""
SageMaker Inference Script for Qwen3-Omni-30B-A3B-Thinking

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
import subprocess
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

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
FFMPEG_PATH = "/usr/local/bin/ffmpeg"


def model_fn(model_dir):
    """Load model - called once when container starts."""
    global model, processor, process_mm_info
    
    logger.info(f"Loading model: {MODEL_PATH}")
    
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
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    
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
                use_audio_in_video=USE_AUDIO_IN_VIDEO
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
