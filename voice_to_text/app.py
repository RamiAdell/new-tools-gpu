from flask import Flask, request, jsonify
import os
import whisper
import ffmpeg
from pydub.utils import mediainfo
from deep_translator import GoogleTranslator
import torch

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
API_KEY = "GPukTcc2FXcAo32U6j6y5rOK8LJW5QAf"


def load_whisper_model():
    try:
        model = whisper.load_model("small", device="cuda" if torch.cuda.is_available() else "cpu")
        print("Whisper model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return None

model = load_whisper_model()


@app.before_request
def authenticate():
    if request.headers.get('X-Api-Key') != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

@app.route('/upload', methods=['POST'])
def upload_media():
    if 'audio' not in request.files:
        return jsonify({"error": "No media file provided"}), 400

    file = request.files['audio']
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    
    filename = f"{user_id}_uploaded_audio.wav"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    if file.mimetype.startswith('video/'):
        try:
            duration = float(mediainfo(file_path)['duration'])
        except Exception:
            os.remove(file_path)
            return jsonify({"error": "Failed to get video duration"}), 400

        if duration > 300:
            os.remove(file_path)
            return jsonify({"error": "File duration exceeds 5 minute limit"}), 400

        
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_id}_extracted_audio.wav")
        if not extract_audio_from_video(file_path, audio_path):
            return jsonify({"error": "Failed to extract audio"}), 500
        os.remove(file_path)  
        file_path = audio_path

    return jsonify({
        "success": True,
        "message": "Media processed successfully",
        "file_path": file_path
    })

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    user_id = data.get('user_id')
    target_lang = data.get('language', 'en')

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    
    possible_files = [
        f"{user_id}_extracted_audio.wav",
        f"{user_id}_uploaded_audio.wav"
    ]
    
    file_path = None
    for filename in possible_files:
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(path):
            file_path = path
            break

    if not file_path:
        return jsonify({"error": "Audio file not found"}), 404

    try:
        
        result = model.transcribe(file_path, task="transcribe", language=None)
        text = result["text"]
        lang = result.get("language", "en")

        if target_lang != lang:
            text = translate_text(text, target_lang)

        return jsonify({
            "success": True,
            "detected_language": lang,
            "text": text
        })

    except Exception as e:
        print(f"Transcription error: {e}")
        return jsonify({"error": str(e)}), 500

def translate_text(text, target_lang):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        return f"Translation error: {e}"

def extract_audio_from_video(video_path, audio_path):
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path,
                   acodec='pcm_s16le',  
                   ar='16000',          
                   ac=1)                
            .run(overwrite_output=True, quiet=True)
        )
        return True
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)