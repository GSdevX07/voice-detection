# 🎙️ AI Voice & Scam Detection API

![Version](https://img.shields.io/badge/version-v2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> A state-of-the-art, multi-modal system designed to protect against AI-generated deepfakes, emotional manipulation, and sophisticated voice scams.

---

## 📖 Project Overview

As digital interactions increasingly shift towards voice-based communication, the risk of sophisticated voice scams (e.g., virtual kidnappings, grandchild scams) has grown exponentially. Modern cybercriminals leverage advanced AI to generate realistic deepfake voices, bypassing traditional security measures.

The **AI Voice & Scam Detection System** is a comprehensive solution that analyzes audio and text inputs in real-time. By combining acoustic anomaly detection with natural language understanding, it provides a multi-layered defense mechanism, resulting in a unified **Fraud Risk Score**.

## ✨ Key Features

*   **Multi-Modal Input:** Supports direct audio file uploads (`.mp3`, `.wav`), raw text input, and continuous real-time **WebSocket audio streaming**.
*   **Voice AI Detection (AASIST):** Evaluates the acoustic properties of the audio to determine the probability of it being AI-generated vs. human.
*   **Real-time Transcription (Whisper):** Accurately transcribes incoming audio into text, with automatic language detection.
*   **Scam Intent Classification (BERT + RAG):** 
    *   A fine-tuned **BERT model** classifies the transcript to detect scam probability and categorization.
    *   A **RAG (Retrieval-Augmented Generation) engine** contextually analyzes the transcript against known scam patterns.
*   **Emotion Analysis:** Extracts the dominant emotion and intensity (e.g., urgency, fear, panic), which are strong indicators of social engineering tactics.
*   **Risk Fusion Engine:** A custom algorithm takes all signals into account to calculate a unified **Fraud Risk Score**, highlighting the primary contributing risk signal (Explainable AI).

---

## 🚀 Getting Started

### 1️⃣ Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/voice-detection.git
cd voice-detection

# Create and activate a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```
*(Ensure `ffmpeg` is installed on your system for audio conversion).*

### 3️⃣ Run the Server

```bash
cd voice-detection-api
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Endpoints

### `POST /api/voice-detection`
Accepts Base-64 encoded audio or raw text.

**Headers:**
`x-api-key: <your_api_key>` (Defaults to: `sk_test_123456789`)

**Request Payload:**
```json
{
  "input_type": "audio",
  "audioFormat": "mp3",
  "audioBase64": "<BASE64_ENCODED_AUDIO>",
  "sensitivity": "NORMAL"
}
```

**Response Payload:**
```json
{
  "language": "English",
  "voice_type": "HUMAN",
  "scam_detected": true,
  "scam_type": "URGENCY",
  "fraud_risk_score": 85,
  "risk_level": "HIGH",
  "primary_risk_signal": "EMOTION",
  "confidence": 0.92
}
```

### `POST /api/voice-detection-upload`
Multipart file upload endpoint for `.mp3` and `.wav` files.

### `WS /ws/stream-detection`
WebSocket endpoint for real-time chunked audio streaming and active intervention.

---

## 🧠 Models Used

1. **AASIST:** Deep anti-spoofing neural network for detecting synthetic / converted speech ([AASIST Paper](https://arxiv.org/abs/2110.01200)).
2. **OpenAI Whisper:** Automatic Speech Recognition (ASR).
3. **Fine-Tuned BERT:** NLP model trained on a custom dataset of scam interactions and fraudulent texts.
4. **DistilBERT Emotion Engine:** Extracts multi-class emotions (fear, panic, urgency).

## 📄 License
This project is released under the **MIT License**.

