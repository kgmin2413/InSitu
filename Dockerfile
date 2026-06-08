# InSitu backend Docker image for RTX 5090 / Blackwell full 3DGS runs.
# Host example: NVIDIA Driver 580.x, nvidia-smi CUDA Version 13.0
# Container stack: CUDA 12.8 + cuDNN devel + PyTorch cu128 + gsplat examples.
# Rationale: RTX 5090 requires sm_120 support; PyTorch cu128 is the working baseline.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Asia/Seoul
ARG PYTORCH_VERSION=2.7.0
ARG TORCHVISION_VERSION=0.22.0
ARG TORCHAUDIO_VERSION=2.7.0
ARG NUMPY_VERSION=1.26.4
ARG OPENCV_VERSION=4.11.0.86

ENV TZ=${TZ} \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="12.0" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# System dependencies for COLMAP, OpenCV, Open3D, gsplat CUDA extensions, and video processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    git \
    git-lfs \
    wget \
    curl \
    ca-certificates \
    pkg-config \
    ffmpeg \
    colmap \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libegl1 \
    libgles2 \
    libgomp1 \
    libboost-all-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libfreeimage-dev \
    libmetis-dev \
    libsqlite3-dev \
    libglew-dev \
    libcgal-dev \
    htop \
    tmux \
    vim \
    nano \
    unzip \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Isolated Python environment.
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --upgrade pip setuptools wheel packaging

# PyTorch cu128: required for RTX 5090 / sm_120 support.
RUN pip install \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/cu128

# Keep the current known-good full 3DGS runtime around numpy 1.x. The gsplat
# example stack requires numpy<2, while newer OpenCV wheels may request numpy>=2.
RUN printf '%s\n' \
    "numpy==${NUMPY_VERSION}" \
    "opencv-python==${OPENCV_VERSION}" \
    "opencv-python-headless==${OPENCV_VERSION}" \
    "rich==14.3.4" \
    > /tmp/insitu_constraints.txt

# Core Python packages used by the InSitu backend pipeline.
RUN pip install -c /tmp/insitu_constraints.txt \
    numpy==${NUMPY_VERSION} \
    scipy \
    scikit-image \
    scikit-learn \
    opencv-python==${OPENCV_VERSION} \
    opencv-python-headless==${OPENCV_VERSION} \
    pillow \
    imageio \
    imageio-ffmpeg \
    tqdm \
    rich==14.3.4 \
    pyyaml \
    pandas \
    matplotlib \
    plyfile \
    open3d \
    trimesh \
    kornia \
    timm \
    einops \
    transparent-background

# Install the checked-out gsplat repo and its example trainer dependencies.
# The repo is copied into /opt so the installed package remains available even
# when docker-compose mounts the development workspace over /workspace/InSitu.
COPY third_party/gsplat /opt/gsplat
RUN find /opt/gsplat -name '*.so' -delete \
    && pip install -c /tmp/insitu_constraints.txt \
      -r /opt/gsplat/examples/requirements.txt \
      --no-build-isolation \
    && pip install -e /opt/gsplat --no-build-isolation

# Build-time sanity check. GPU availability is checked at runtime, not during image build.
RUN python - <<'PY'
import cv2
import gsplat
import numpy
import torch
import tyro
import viser
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('numpy:', numpy.__version__)
print('opencv:', cv2.__version__)
print('gsplat:', gsplat.__version__)
print('tyro:', tyro.__version__)
print('viser:', viser.__version__)
PY

WORKDIR /workspace/InSitu

# The repository is mounted by docker-compose for development. This image already
# includes COLMAP, PyTorch cu128, gsplat, and gsplat example dependencies, so the
# full chair_marker_gsplat path can run without post-start dependency installs.
CMD ["/bin/bash"]
