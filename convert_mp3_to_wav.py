import os
from pydub import AudioSegment

INPUT_ROOT = "train_data_raw_mp3"
OUTPUT_ROOT = "train_data"
TARGET_SR = 16000

for root, _, files in os.walk(INPUT_ROOT):
    for file in files:
        if not file.lower().endswith(".mp3"):
            continue

        src_path = os.path.join(root, file)

        rel_path = os.path.relpath(root, INPUT_ROOT)
        dst_dir = os.path.join(OUTPUT_ROOT, rel_path)
        os.makedirs(dst_dir, exist_ok=True)

        dst_path = os.path.join(dst_dir, file.replace(".mp3", ".wav"))

        audio = AudioSegment.from_mp3(src_path)
        audio = audio.set_frame_rate(TARGET_SR)
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)  # 16-bit PCM

        audio.export(dst_path, format="wav")
        print("Converted:", dst_path)
