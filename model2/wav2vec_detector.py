import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model


class Wav2Vec2DeepfakeDetector:
    def __init__(self, device="cpu"):
        self.device = device

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        ).to(device)
        self.model.eval()

    def predict(self, wav_path):
        # ✅ LOAD AUDIO WITH LIBROSA (NO TORCHAUDIO, NO FFmpeg)
        speech, sr = librosa.load(wav_path, sr=16000, mono=True)

        inputs = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)

        hidden_states = outputs.last_hidden_state

        # Simple heuristic score (variance-based)
        score = torch.var(hidden_states).item()

        return score
