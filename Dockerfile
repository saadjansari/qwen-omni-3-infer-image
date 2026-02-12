# Dockerfile for Qwen3-Omni-30B-A3B-Thinking SageMaker Endpoint

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    xz-utils \
    libsndfile1 \
    ninja-build \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Install static ffmpeg
RUN curl -L -o /tmp/ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp \
    && cp /tmp/ffmpeg-*/ffmpeg /usr/local/bin/ \
    && cp /tmp/ffmpeg-*/ffprobe /usr/local/bin/ \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch (match CUDA version)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers (pinned to 4.57.6 — Qwen3-Omni support + compatible MoE expert structure for LoRA)
RUN pip install transformers==4.57.6
# Install transformers from source (REQUIRED for Qwen3-Omni support)
#RUN pip install git+https://github.com/huggingface/transformers

# Install other dependencies
RUN pip install \
    accelerate \
    "qwen-omni-utils[decord]" \
    soundfile \
    boto3 \
    flask \
    gunicorn \
    peft

# Install flash attention (optional - improves performance)
RUN pip install packaging && \
    pip install flash-attn --no-build-isolation || echo "Flash attention install failed, will use eager attention"

# Create cache directories
RUN mkdir -p /tmp/hf_cache /opt/ml/model
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

# Copy inference code
WORKDIR /opt/ml/code
COPY inference.py /opt/ml/code/
COPY serve.py /opt/ml/code/

# Make serve.py executable
RUN chmod +x /opt/ml/code/serve.py

# Expose port (SageMaker uses 8080)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

# Entry point
ENTRYPOINT ["python", "/opt/ml/code/serve.py"]
