import librosa
import numpy as np
import joblib
import sys

MODEL_PATH = "voice_ai_detector_rf.pkl"

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000, duration=5)
    
    mfcc = np.mean(
        librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13),
        axis=1
    )
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    spec_flat = np.mean(librosa.feature.spectral_flatness(y=audio))

    return np.hstack([mfcc, zcr, spec_flat])

# load trained model
model = joblib.load(MODEL_PATH)

def predict(audio_path):
    feats = extract_features(audio_path).reshape(1, -1)
    probs = model.predict_proba(feats)[0]

    label = "AI_GENERATED" if probs[1] > probs[0] else "HUMAN"
    confidence = float(max(probs))

    print("Prediction:", label)
    print("Confidence:", round(confidence, 2))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python model2_infer.py <audio.wav>")
        exit(1)

    predict(sys.argv[1])
