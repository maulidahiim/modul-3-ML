import os

os.environ["PATH"] += os.pathsep + r"C:\Users\ASUS\Downloads\ffmpeg-8.1.1-essentials_build\ffmpeg-8.1.1-essentials_build\bin"

import whisper

model = whisper.load_model("base")

result = model.transcribe(
    "audio/my_speech.wav",
    language="id"
)

print(result["text"])

with open("output/transcription.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

#awal nya saya menggunakan tiny dan hasil dari audionya tidak seberapa akurat pada akhirnya saya menggantinya menggunakan base dan terbukti hasilnya lebih akurat