# 🎙️ AI Voice Detection API (Deepfake vs Human)

This project is a **REST API–based AI Voice Detection system** that detects whether a given audio sample is **AI-generated or Human speech**.  
It also performs **language detection** and returns a structured JSON output as required by the problem statement.

The system is built using:
- **AASIST** (state-of-the-art audio anti-spoofing model)
- **Ensemble Learning** (AASIST + secondary ML model)
- **Whisper** for language detection
- **FastAPI** for REST API deployment

---

## 🚀 Features

- ✅ Accepts **Base64 encoded MP3 audio**
- ✅ Automatically converts MP3 → WAV
- ✅ Detects **spoken language**
- ✅ Classifies audio as **AI_GENERATED / HUMAN**
- ✅ Provides **confidence score**
- ✅ Returns **model insights**
- ✅ Fully exposed via **REST API endpoint**

---

## 📥 Input Format (API Request)

**Endpoint**
POST /api/voice-detection



**Headers**
x-api-key: sk_test_123456789
Content-Type: application/json



**Request Body**
```json
{
  "language": "auto",
  "audioFormat": "mp3",
  "audioBase64": "<BASE64_ENCODED_AUDIO>"
}
📤 Output Format (API Response)
{
  "language": "English",
  "final_classification": "AI_GENERATED",
  "confidenceScore": 0.96,
  "modelInsights": [
    "high pitch consistency",
    "lack of natural pauses",
    "vocoder artifacts detected"
  ]
}


⚙️ How to Run the Project (Judges)
1️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run API Server
cd aasist
uvicorn api:app --host 0.0.0.0 --port 8000
4️⃣ Open API Docs
http://127.0.0.1:8000/docs


🧠 Models Used
🔹 AASIST
Deep anti-spoofing neural network
Detects synthetic / converted speech

🔹 Secondary Model (Model-2)
Trained on mixed human & AI samples
Adds robustness via ensemble decision

🔹 Language Detection
Powered by OpenAI Whisper
Supports: English, Hindi, Tamil, Telugu, Malayalam



🗂️ Project Structure
voice-detection/
│
├── aasist/
│   ├── api.py                # FastAPI endpoint
│   ├── infer_single.py       # Inference pipeline
│   ├── ensemble_detector.py # Ensemble logic
│   ├── models/               # AASIST weights
│
├── model2/
│   └── wav2vec_detector.py   # Secondary model
│
├── convert_mp3_to_wav.py
├── requirements.txt
├── README.md


🔐 API Key Note
A demo API key is used for hackathon testing:
sk_test_123456789
In production, this should be stored securely (env variable).


📚 Acknowledgements & Research Credit
This project is built on top of the following research work:

AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
https://arxiv.org/abs/2110.01200

Original AASIST repository and ASVspoof datasets were used for model design inspiration and benchmarking.

