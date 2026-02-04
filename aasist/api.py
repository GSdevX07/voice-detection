from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import os
import uuid
import sys
import subprocess

def convert_mp3_to_wav(mp3_path, wav_path):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", mp3_path,
            "-ac", "1",
            "-ar", "16000",
            wav_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


# ---------- PATH FIX (VERY IMPORTANT) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # aasist/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # voice-detection/
sys.path.append(PROJECT_ROOT)

# ---------- IMPORTS ----------
from infer_single import predict

# ---------- CONFIG ----------
API_KEY = os.getenv("API_KEY")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="AI Voice Detection API",
    version="1.0"
)

# ---------- REQUEST SCHEMA ----------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ---------- HELPERS ----------
def save_base64_audio(audio_base64: str, audio_format: str) -> str:
    """Decode Base64 audio and save to file"""
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{audio_format}")

    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    return file_path


def prepare_audio(audio_base64: str, audio_format: str) -> str:
    """Base64 → MP3 → WAV"""
    if audio_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    mp3_path = save_base64_audio(audio_base64, "mp3")
    wav_path = mp3_path.replace(".mp3", ".wav")

    convert_mp3_to_wav(mp3_path, wav_path)

    return wav_path

# ---------- API ENDPOINT ----------
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

    # 🤖 Run detection pipeline
    result = predict(wav_path)

    return result

# ---------- ROOT (FOR HEALTH CHECK) ----------
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "AI Voice Detection API is running",
        "endpoint": "/api/voice-detection"
    }
