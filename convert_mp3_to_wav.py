import os
from pydub import AudioSegment

TARGET_SR = 16000

def convert_mp3_to_wav(input_path: str, output_path: str):
    """
    Converts a single MP3 file to WAV (16kHz, mono, 16-bit PCM)
    """
    audio = AudioSegment.from_mp3(input_path)
    audio = audio.set_frame_rate(TARGET_SR)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)  # 16-bit PCM

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio.export(output_path, format="wav")

    return output_path


if __name__ == "__main__":
    # standalone test
    inp = "test.mp3"
    out = "test.wav"
    convert_mp3_to_wav(inp, out)
    print("Converted:", out)
