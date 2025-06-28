FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app


RUN apt-get update && apt-get install -y \
    git git-lfs wget curl ffmpeg \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    python3 python3-pip python-is-python3 \
    && apt-get clean


COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


RUN git lfs install && git clone https://huggingface.co/goofyai/3d_render_style_xl /app/3d_render_style_xl


COPY . .

EXPOSE 5006


CMD ["gunicorn", "--bind", "0.0.0.0:5006", "--workers=1", "--threads=4", "app:app"]
