FROM nvidia/cuda:12.1.1-base

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
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

    # Set up Python environment
RUN python3.10 -m pip install --upgrade pip
RUN pip install \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install ONNX Runtime with CUDA 12 support
RUN pip uninstall onnxruntime onnxruntime-gpu -y && \
    pip install onnxruntime-gpu==1.16.3 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install onnxruntime-gpu

RUN python3 -c "from rembg import new_session; session = new_session('u2net', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']); print('Model downloaded successfully')"

COPY . /app

RUN mkdir -p /tmp/uploads /tmp/processed
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV CUDA_VISIBLE_DEVICES=0
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 5550


# Use gunicorn with proper logging and GPU-friendly settings
CMD ["gunicorn", "--bind", "0.0.0.0:5550", "--workers", "1", "--threads", "4", "--timeout", "1800", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "app:app"]