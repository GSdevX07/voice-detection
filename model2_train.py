import os
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "train_data"
SAMPLE_RATE = 16000

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    spec_flat = np.mean(librosa.feature.spectral_flatness(y=audio))
    return np.hstack([mfcc, zcr, spec_flat])

X, y = [], []

for label, cls in enumerate(["human", "ai"]):
    class_path = os.path.join(DATA_DIR, cls)

    for lang in os.listdir(class_path):
        lang_path = os.path.join(class_path, lang)

        for file in os.listdir(lang_path):
            if file.endswith(".wav"):
                feats = extract_features(os.path.join(lang_path, file))
                X.append(feats)
                y.append(label)

X = np.array(X)
y = np.array(y)

print("Training samples:", len(X))

model = RandomForestClassifier(n_estimators=300)
model.fit(X, y)

joblib.dump(model, "voice_ai_detector_rf.pkl")
print("✅ Model trained and saved as voice_ai_detector_rf.pkl")
