FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /workspace/code

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch CUDA wheels are installed explicitly. TensorRT engine export may still
# require a host/container TensorRT setup compatible with the target GPU.
RUN python3 -m pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

RUN python3 -m pip install \
    onnx \
    onnxruntime-gpu \
    tensorrt

CMD ["bash"]
