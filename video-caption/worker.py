# worker.py
import torch
from whisper import load_model

class GPUWorker:
    def __init__(self):
        self._initialize_device()
        self.model = None
        
    def _initialize_device(self):
        """Initialize CUDA context safely"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            # Critical: Initialize CUDA context immediately
            torch.cuda.init()
            torch.cuda.set_device(0)
            # Warm-up CUDA
            torch.randn(1, device='cuda')
            
    def load_model(self):
        if self.model is None:
            try:
                if self.device == "cuda":
                    self.model = load_model("small", device="cuda")
                    # Verify model is on GPU
                    assert next(self.model.parameters()).device.type == 'cuda'
                else:
                    self.model = load_model("small", device="cpu")
            except Exception as e:
                raise RuntimeError(f"Model load failed: {str(e)}")

    def transcribe(self, audio_path):
        if self.model is None:
            self.load_model()
        return self.model.transcribe(audio_path)