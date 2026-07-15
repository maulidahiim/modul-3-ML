"""Lab 3.5 -- Voice Cloning with Qwen3-TTS

NOTE on adaptation from the module's listed code:
The module's Section 8.4 code (`Qwen2_5OmniForConditionalGeneration`, model id
"Qwen/Qwen3-TTS") does not match the real released model/API. The actual model
family on HuggingFace is Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-{Base,CustomVoice,
VoiceDesign}, loaded via the official `qwen-tts` pip package's `Qwen3TTSModel`
wrapper. Voice cloning specifically requires the "Base" checkpoint and its
`generate_voice_clone()` method. This script uses that real, working API
(installed in a separate Python 3.11 venv, since the package's pinned
`accelerate==1.12.0` requires Python >=3.10) to perform the same task as the
module: zero-shot voice cloning from a short reference clip.

Reference audio here is the user's own real voice recording (alvin_take2.wav),
consistent with the lab's guidance to clone your own voice when in doubt
about third-party consent. Both output files below use the same English
text/language, so the two cloned samples are directly comparable.

We use x_vector_only_mode=True (speaker-embedding-only cloning) rather than
ICL mode, since it only needs the reference audio -- no transcript of the
reference audio is required, which fits an arbitrary spoken-out-loud clip.
"""
import os
import torch
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

REFERENCE_AUDIO = "audio/my_speech.wav"

CLONE_LANGUAGE = "English"

CLONE_TEXT = (
    "Hello, this is my voice cloning experiment "
    "using artificial intelligence technology."
)

OUTPUT_PATHS = [
    "output/voice_clone.wav"
]

# NOTE: partial .to("mps") moves left some qwen-tts submodules (e.g. the
# talker's text embedding) on CPU while inputs were on MPS, causing
# "Placeholder storage has not been allocated on MPS device!". The model is
# only 0.6B params, so we run it on CPU for reliability instead.
DEVICE = "cpu"
print(f"Using device: {DEVICE}")


def main():
    print(f"Loading {MODEL_ID} ... (first run downloads the model weights)")
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, dtype=torch.float32)

    print(f"Cloning voice from reference: {REFERENCE_AUDIO}")
    wavs, sample_rate = model.generate_voice_clone(
        text=CLONE_TEXT,
        language=CLONE_LANGUAGE,
        ref_audio=REFERENCE_AUDIO,
        x_vector_only_mode=True,  # speaker-embedding-only: no reference transcript needed
    )

    audio = wavs[0]
    for out_path in OUTPUT_PATHS:
        sf.write(out_path, audio, sample_rate)
        print(f"Saved cloned voice to: {out_path}")

    # Plot reference vs cloned waveform for the deliverable
    ref_data, ref_sr = sf.read(REFERENCE_AUDIO)
    cloned_data, cloned_sr = sf.read(OUTPUT_PATHS[0])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(np.linspace(0, len(ref_data) / ref_sr, len(ref_data)), ref_data, linewidth=0.5)
    axes[0].set_title("Reference Audio Waveform (alvin_take2.wav)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(np.linspace(0, len(cloned_data) / cloned_sr, len(cloned_data)), cloned_data, linewidth=0.5, color="orange")
    axes[1].set_title("Qwen3-TTS Cloned Output Waveform")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig("lab3_5_waveform_comparison.png", dpi=150)
    print("Saved: lab3_5_waveform_comparison.png")


if __name__ == "__main__":
    main()
