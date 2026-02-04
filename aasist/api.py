from fastapi import FastAPI, UploadFile, File
import os
import shutil
import subprocess

from infer_single import predict

# 🚨 FORCE DOCS ENABLED
app = FastAPI(
    title="AI Voice Detection API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def convert_to_wav(input_path):
    wav_path = input_path.rsplit(".", 1)[0] + ".wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", "16000",
            wav_path
        ],
        check=True
    )
    return wav_path


@app.get("/")
def root():
    return {"status": "API running"}


@app.post("/analyze-voice")
async def analyze_voice(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if not input_path.endswith(".wav"):
        wav_path = convert_to_wav(input_path)
    else:
        wav_path = input_path

    return predict(wav_path)
