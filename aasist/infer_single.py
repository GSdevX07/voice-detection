import sys
import os
import json

# ---------- BASE PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # aasist/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # voice-detection/

# ---------- PATH FIXES ----------
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(BASE_DIR, "models"))

# ---------- IMPORTS ----------
import torch
import librosa
import numpy as np
import whisper

from AASIST import Model
from data_utils import pad
from ensemble_detector import ensemble_decision
from model2.wav2vec_detector import Wav2Vec2DeepfakeDetector

# ---------- CONFIG ----------
CONFIG_PATH = os.path.join(BASE_DIR, "config", "AASIST.conf")

# ✅ FIXED PATH (THIS WAS THE BUG)
MODEL_PATH = os.path.join(BASE_DIR, "models", "weights", "AASIST.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000

# ---------- LOAD WHISPER (ONCE) ----------
whisper_model = whisper.load_model("base")

LANG_MAP = {
    "ta": "Tamil",
    "te": "Telugu",
    "hi": "Hindi",
    "ml": "Malayalam",
    "en": "English"
}

# ---------- LANGUAGE DETECTION ----------
def detect_language(audio_path: str) -> str:
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    _, probs = whisper_model.detect_language(mel)

    lang_code = max(probs, key=probs.get)
    return LANG_MAP.get(lang_code, "Unknown")

# ---------- LOAD AASIST (CACHED) ----------
_AASIST_MODEL = None

def load_model():
    global _AASIST_MODEL
    if _AASIST_MODEL is not None:
        return _AASIST_MODEL

    with open(CONFIG_PATH, "r") as f:
        cfg = eval(f.read())  # expected AASIST format

    d_args = cfg["model_config"]
    model = Model(d_args)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)

    model.to(DEVICE)
    model.eval()

    _AASIST_MODEL = model
    return model

# ---------- AUDIO PREPROCESS ----------
def preprocess_audio(audio_path: str) -> torch.Tensor:
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    audio = pad(audio, SAMPLE_RATE * 4)  # 4 seconds
    audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return audio

# ---------- MAIN PREDICT ----------
def predict(audio_path: str) -> dict:
    # 🔹 Language detection
    language = detect_language(audio_path)

    # 🔹 AASIST
    aasist_model = load_model()
    audio_tensor = preprocess_audio(audio_path)

    with torch.no_grad():
        _, output = aasist_model(audio_tensor)

    aasist_score = (output[0][1] - output[0][0]).item()

    # 🔹 Model 2 (Wav2Vec2)
    wav2vec_model = Wav2Vec2DeepfakeDetector()
    wav2vec_score = wav2vec_model.predict(audio_path)

    # 🔹 Ensemble
    final_label, confidence, insights = ensemble_decision(
        aasist_score,
        wav2vec_score
    )

    # 🔹 FINAL JSON
    result = {
        "language": language,
        "final_classification": final_label,
        "confidenceScore": round(confidence, 2),
        "modelInsights": insights
    }

    return result

# ---------- CLI ----------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_single.py <audio.mp3|audio.wav>")
        sys.exit(1)

    out = predict(sys.argv[1])
    print(json.dumps(out, indent=2))
