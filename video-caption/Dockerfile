FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive


WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    ffmpeg \
    build-essential \
    wget \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python


RUN python3 -m pip install --upgrade pip


COPY requirements.txt .


RUN pip install --no-cache-dir torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121


RUN pip install --no-cache-dir -r requirements.txt


RUN pip install --no-cache-dir openai-whisper


COPY . /app


RUN mkdir -p /app/fonts /app/.cache/torch /app/.cache/whisper


COPY ./fonts/ /app/fonts/


ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5003
ENV PYTHONUNBUFFERED=1
ENV FONT_FOLDER=/app/fonts


ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0
ENV NCCL_P2P_DISABLE=1
ENV TOKENIZERS_PARALLELISM=false


ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV CUDA_LAUNCH_BLOCKING=1


ENV WHISPER_MODEL_SIZE=small
ENV TORCH_HOME=/app/.cache/torch


RUN chmod -R 755 /app


EXPOSE 5003


HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5003/health || exit 1



CMD ["python", "-c", "import multiprocessing; multiprocessing.set_start_method('spawn', force=True); import app; app.app.run(host='0.0.0.0', port=5003, threaded=True)"]