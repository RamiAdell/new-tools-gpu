FROM python:3.9-slim-buster


RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN python3 -c "import whisper; whisper.load_model('small')"

COPY app.py .


ENV WHISPER_MODEL="/root/.cache/whisper/small.pt"
ENV WHISPER_HUGGINGFACE_WHISPER_CACHE=1

EXPOSE 5005


CMD ["gunicorn", "--bind", "0.0.0.0:5005", "--timeout", "300", "--workers=1", "app:app"]