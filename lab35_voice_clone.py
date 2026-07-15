import os
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("output", exist_ok=True)

print(f"Using device : {DEVICE}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Qwen3-TTS model...\n")


model = Qwen3TTSModel.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)


print("Model loaded successfully")


# ============================================================
# VOICE CLONING FUNCTION
# ============================================================

def clone_voice(
    text,
    reference_audio_path,
    output_path="output/voice_clone.wav"
):

    print("\nGenerating cloned voice...")


    # Generate voice cloning
    wavs, sample_rate = model.generate_voice_clone(
        text=text,

        # audio contoh suara
        ref_audio=reference_audio_path,

        # isi kalimat yang ada pada audio referensi
        # SESUAIKAN jika isi audio berbeda
        ref_text="Hello, this is my reference voice sample.",

        language="english",

        # gunakan mode cloning lengkap
        x_vector_only_mode=False
    )


    # Ambil hasil audio
    audio = wavs[0]


    # Tensor -> numpy
    if isinstance(audio, torch.Tensor):
        audio = (
            audio
            .detach()
            .cpu()
            .numpy()
        )


    # Simpan wav
    sf.write(
        output_path,
        audio,
        sample_rate
    )


    print("\n==============================")
    print("VOICE CLONING FINISHED")
    print("==============================")
    print(f"Saved : {output_path}")



# ============================================================
# RUN
# ============================================================

# Kalimat hasil suara AI
text_input = (
    "Hello, this is a voice cloning experiment "
    "using artificial intelligence technology. "
    "I am testing the ability of AI to imitate my voice."
)


clone_voice(
    text=text_input,
    reference_audio_path="audio/my_speech.wav",
    output_path="output/voice_clone.wav"
)