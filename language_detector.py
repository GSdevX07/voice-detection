import whisper

# load once (slow model not needed)
model = whisper.load_model("base")

def detect_language(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)

    # get highest probability language
    lang_code = max(probs, key=probs.get)

    LANG_MAP = {
        "ta": "Tamil",
        "te": "Telugu",
        "hi": "Hindi",
        "ml": "Malayalam",
        "en": "English"
    }

    return LANG_MAP.get(lang_code, "Unknown")
