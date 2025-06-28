import os
import uuid
import torch
from flask import Flask, request, jsonify, send_file
from deep_translator import GoogleTranslator
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_FOLDER = os.path.join(BASE_DIR, "static", "processed")
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)


app = Flask(__name__)
app.config.from_object(Config)

print("Loading model...")
model_id = "goofyai/3d_render_style_xl"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision="fp16"
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
print("Model loaded and optimized.")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        if any('\u0600' <= c <= '\u06ff' for c in prompt):
            prompt = GoogleTranslator(source="auto", target="en").translate(prompt)
    except Exception as e:
        return jsonify({"error": f"Translation error: {str(e)}"}), 500

    try:
        result = pipe(prompt)
        image = result.images[0]

        filename = f"{uuid.uuid4().hex[:8]}.png"
        file_path = os.path.join(Config.PROCESSED_FOLDER, filename)
        image.save(file_path, "PNG")

        return send_file(file_path, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5006, debug=True)
