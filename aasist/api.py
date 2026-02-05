import os
import sys
import base64
import uuid
import subprocess

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

# ---------- PATH FIX ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # aasist/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # voice-detection/
sys.path.insert(0, PROJECT_ROOT)

# ---------- IMPORT MODEL ----------
from aasist.infer_single import predict

# ---------- CONFIG ----------
API_KEY = os.getenv("API_KEY")  # must be set in environment
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="AI Voice Detection API",
    version="1.0"
)

# ---------- REQUEST SCHEMA ----------
class VoiceRequest(BaseModel):
    language: Optional[str] = "auto"
    audioFormat: str
    audioBase64: str

# ---------- AUDIO HELPERS ----------
def save_base64_audio(audio_base64: str, ext: str) -> str:
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    file_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")

    with open(path, "wb") as f:
        f.write(audio_bytes)

    return path


def convert_mp3_to_wav(mp3_path: str) -> str:
    wav_path = mp3_path.replace(".mp3", ".wav")

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", mp3_path,
                "-ac", "1",
                "-ar", "16000",
                wav_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except Exception:
        raise HTTPException(status_code=500, detail="FFmpeg conversion failed")

    return wav_path


def prepare_audio(audio_base64: str, audio_format: str) -> str:
    if audio_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    mp3_path = save_base64_audio(audio_base64, "mp3")
    wav_path = convert_mp3_to_wav(mp3_path)

    return wav_path

# ---------- API ----------
@app.post("/api/voice-detection")
def analyze_voice(
    data: VoiceRequest,
    x_api_key: str = Header(...)
):
    # 🔐 API KEY CHECK
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 🎧 Prepare audio
    wav_path = prepare_audio(data.audioBase64, data.audioFormat)

    # 🤖 Predict
    result = predict(wav_path)

    return result

# ---------- HEALTH ----------
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "AI Voice Detection API is running",
        "endpoint": "/api/voice-detection"
    }

