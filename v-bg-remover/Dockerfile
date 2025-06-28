FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    g++ \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgomp1 \
    libx264-dev \
    x264 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install onnxruntime-gpu

RUN python3 -c "from rembg import new_session; session = new_session('u2net', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']); print('Model downloaded successfully')"

COPY . /app

RUN mkdir -p /tmp/uploads /tmp/processed

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 5550


# Use gunicorn with proper logging and GPU-friendly settings
CMD ["gunicorn", "--bind", "0.0.0.0:5550", "--workers", "1", "--threads", "4", "--timeout", "1800", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "app:app"]