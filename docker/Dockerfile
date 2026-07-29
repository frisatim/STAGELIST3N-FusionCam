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

# Versions alignees sur l'environnement de reference du stage
# (voir docs/ENVIRONNEMENT_REFERENCE.md et requirements.lock.txt).
# Les wheels PyTorch cu128 embarquent leurs bibliotheques CUDA ; l'image de
# base fournit cuDNN et le runtime. L'export d'engine TensorRT reste lie au
# GPU cible.
RUN python3 -m pip install \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

RUN python3 -m pip install \
    onnx==1.20.1 \
    onnxruntime-gpu==1.24.1 \
    tensorrt==10.16.0.72

CMD ["bash"]
