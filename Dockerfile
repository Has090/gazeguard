FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgl1 / libglib2.0-0: needed by OpenCV and MediaPipe; git: needed by DVC
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (much smaller than the default CUDA build), then the rest
RUN pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x scripts/run_all.sh

# One command reproduces the registered metrics:
#   docker run --rm --env-file .env gazeguard-train
CMD ["bash", "scripts/run_all.sh"]
