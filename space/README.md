---
title: N-ATLAS Hotline
emoji: 📞
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
python_version: "3.11"
---

# N-ATLAS Hotline (evaluator demo)

Voice-first assistant for Nigerians. Speak in **English, Yoruba, Hausa, or Igbo**.

- **Hear (official N-ATLaS ASR):** `NCAIR1/NigerianAccentedEnglish`, `NCAIR1/Yoruba-ASR`, `NCAIR1/Hausa-ASR`, `NCAIR1/Igbo-ASR`
- **Answer (official N-ATLaS LLM):** `NCAIR1/N-ATLaS` (`from_pretrained` + `apply_chat_template`, as on the model card)
- **Speak (not N-ATLaS):** NCAIR did not release TTS. Yoruba / Igbo / Hausa playback uses Soro-TTS MMS/VITS (`Shinzmann/soro-tts-yor`, `soro-tts-ibo`, `soro-tts-hau`, CC BY-NC). English playback uses gTTS.

Tone marks: if N-ATLaS writes Yoruba or Igbo without diacritics, Soro-TTS can sound flat. That is the LLM text, not a second N-ATLaS model.

Runs on **ZeroGPU**. **Sign in with a Hugging Face account** on the Space (free accounts get more daily GPU than a guest). First request is slow (weights load). Later requests are faster. If you see “ZeroGPU quota exceeded”, wait for the reset or retry while signed in — that is daily GPU time, not an app crash.

N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, powered by Awarri Technologies.
