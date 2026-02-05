import os
import subprocess
import imageio_ffmpeg

TARGET_SR = 16000

def convert_mp3_to_wav(input_path: str, output_path: str):
    """
    Converts MP3 to WAV (16kHz, mono, 16-bit PCM) using bundled ffmpeg
    (Render-compatible)
    """

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", str(TARGET_SR),
            "-sample_fmt", "s16",
            output_path
        ],
        check=True
    )

    return output_path


if __name__ == "__main__":
    inp = "test.mp3"
    out = "test.wav"
    convert_mp3_to_wav(inp, out)
    print("Converted:", out)

