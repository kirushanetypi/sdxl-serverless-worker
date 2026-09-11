# Custom SDXL/Illustrious worker image for RunPod Serverless.
#
# Built by GitHub Actions (.github/workflows/build.yml) and published to GHCR:
#   ghcr.io/<owner>/sdxl-serverless-worker:latest
# The VPS has ~7 GB free disk, so the CUDA image is never built locally.
#
# Base image is the CUDA *runtime* image used by the upstream RunPod worker
# (github.com/runpod-workers/worker-sdxl) - verified to exist on Docker Hub:
# https://hub.docker.com/r/nvidia/cuda/tags?name=12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python 3.11 from deadsnakes (same approach as the upstream worker image).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl libgomp1 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3 /tmp/get-pip.py \
    && rm -f /tmp/get-pip.py \
    && python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /requirements.txt
RUN python3 -m pip install -r /requirements.txt

WORKDIR /app
COPY pipeline_utils.py model_store.py handler.py /app/

# MODEL_ROOT points at the network volume mounted by RunPod Serverless
# (docs.runpod.io/storage/network-volumes: "mount at /runpod-volume").
# HF_HOME stays on the container disk: the weights are downloaded explicitly
# into the volume store, so nothing else needs to be persisted.
ENV MODEL_ROOT=/runpod-volume/models \
    HF_HOME=/tmp/hf-home \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEFAULT_STEPS=30 \
    MAX_CACHED_PIPELINES=2 \
    ALLOW_DOWNLOAD=1

CMD ["python3", "-u", "/app/handler.py"]
