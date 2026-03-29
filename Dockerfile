FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set up Environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_NO_USAGE_STATS=1

# Install system dependencies
RUN apt-get update -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    ffmpeg \
    unzip \
    sox \
    libsox-dev \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libcudnn8=8.* \
    libcudnn8-dev=8.* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV PYTHONPATH="/workspace:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"

# Install pynini separately to avoid potential issues with other dependencies
RUN python3 -m pip install --no-cache-dir pynini==2.1.5

# Install Python dependencies for CosyVoice
COPY CosyVoice/requirements.txt /workspace/CosyVoice/requirements.txt
RUN cd /workspace/CosyVoice && \
    python3 -m pip install -r requirements.txt --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# Install vLLM runtime
RUN python3 -m pip install --no-cache-dir \
    vllm==0.11.0 \
    transformers==4.57.1 \
    numpy==1.26.4 \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# Install Flashinfer
RUN python3 -m pip install --no-cache-dir --only-binary=:all: flashinfer-python

# Install Python dependencies for app server
COPY requirements.simple.txt /workspace/requirements.simple.txt
RUN cd /workspace && \
    python3 -m pip install -r requirements.simple.txt --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# Optional cache-bust arg for source-copy stage only
# How to use: `docker build --build-arg CACHEBUST=$(date +%s) -t cosyvoice:latest .`
ARG CACHEBUST=1
RUN echo "cachebust=${CACHEBUST}" > /dev/null

# Copy local source code after dependency installation for faster rebuilds
COPY CosyVoice/ /workspace/CosyVoice/
COPY app/ /workspace/app/
COPY assets/ /workspace/assets/

# Expose the port for the API
EXPOSE 8000

CMD ["python", "-m", "app.main"]
