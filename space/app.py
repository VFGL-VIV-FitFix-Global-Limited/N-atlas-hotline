"""N-ATLAS Hotline — Gradio Space for evaluators.

Do not call .cuda() at import time. ZeroGPU only attaches CUDA inside
functions decorated with @spaces.GPU.
"""

from __future__ import annotations

import os
import tempfile
import wave
from datetime import datetime, timezone

import gradio as gr
import numpy as np
import spaces
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, VitsModel, pipeline

if os.getenv("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

ASR_IDS = {
    "en": "NCAIR1/NigerianAccentedEnglish",
    "yo": "NCAIR1/Yoruba-ASR",
    "ha": "NCAIR1/Hausa-ASR",
    "ig": "NCAIR1/Igbo-ASR",
}
LANGUAGE_NAMES = {
    "en": "English",
    "yo": "Yoruba",
    "ha": "Hausa",
    "ig": "Igbo",
}
LLM_ID = "NCAIR1/N-ATLaS"
# Official N-ATLaS has no TTS. Soro MMS/VITS speaks Yoruba/Igbo/Hausa (not NCAIR).
TTS_IDS = {
    "yo": "Shinzmann/soro-tts-yor",
    "ha": "Shinzmann/soro-tts-hau",
    "ig": "Shinzmann/soro-tts-ibo",
}

SYSTEM_PROMPT = """You are N-ATLAS Hotline, a helpful voice assistant for Nigerians.
Always respond in {language}. Keep it under 150 words, spoken-style, warm and clear.
For health: general guidance only, advise seeing a doctor for serious symptoms.
For government: general information only, not legal advice.
"""

_asr = {}
_tts = {}
_tok = None
_llm = None


def _ensure_models(lang: str) -> None:
    global _tok, _llm
    if lang not in _asr:
        _asr[lang] = pipeline(
            "automatic-speech-recognition",
            model=ASR_IDS[lang],
            token=os.getenv("HF_TOKEN"),
            device=0,
        )
    if _llm is None:
        _tok = AutoTokenizer.from_pretrained(LLM_ID, token=os.getenv("HF_TOKEN"))
        _llm = AutoModelForCausalLM.from_pretrained(
            LLM_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.getenv("HF_TOKEN"),
        )
        _llm.eval()


def _ensure_tts(lang: str) -> None:
    if lang in _tts or lang not in TTS_IDS:
        return
    model = VitsModel.from_pretrained(TTS_IDS[lang]).to("cpu").eval()
    tokenizer = AutoTokenizer.from_pretrained(TTS_IDS[lang])
    _tts[lang] = (model, tokenizer)


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(np.asarray(audio).squeeze() * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def _speak(text: str, lang: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        if lang == "en":
            from gtts import gTTS

            path = tempfile.mktemp(suffix=".mp3")
            gTTS(text=text, lang="en", tld="com").save(path)
            return path

        _ensure_tts(lang)
        model, tokenizer = _tts[lang]
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            waveform = model(**inputs).waveform[0].cpu().numpy()
        path = tempfile.mktemp(suffix=".wav")
        _write_wav(path, waveform, model.config.sampling_rate)
        return path
    except Exception:
        return None


@spaces.GPU(duration=80)
def _run_on_gpu(audio_path: str, language: str) -> tuple[str, str, str | None]:
    if not audio_path:
        return "", "Record or upload a short voice note first.", None

    lang = {
        "English": "en",
        "Yoruba": "yo",
        "Hausa": "ha",
        "Igbo": "ig",
        "en": "en",
        "yo": "yo",
        "ha": "ha",
        "ig": "ig",
    }.get(language or "English", "en")
    _ensure_models(lang)

    asr_out = _asr[lang](audio_path)
    transcript = asr_out["text"] if isinstance(asr_out, dict) else str(asr_out)

    lang_name = LANGUAGE_NAMES[lang]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=lang_name)},
        {"role": "user", "content": transcript},
    ]
    date_string = datetime.now(timezone.utc).strftime("%d %b %Y")
    prompt = _tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        date_string=date_string,
    )
    inputs = _tok(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(_llm.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = _llm.generate(
            **inputs,
            max_new_tokens=300,
            use_cache=True,
            repetition_penalty=1.12,
            temperature=0.1,
        )
    new = out[0][inputs["input_ids"].shape[1] :]
    reply = _tok.decode(new, skip_special_tokens=True).strip()
    audio_out = _speak(reply, lang)
    return transcript, reply, audio_out


def _log_to_hotline(transcript: str, reply: str, language: str) -> None:
    """POST Q&A to Railway so Space tests land in the same downloadable log."""
    url = (os.getenv("HOTLINE_LOG_URL") or "").rstrip("/")
    key = os.getenv("HOTLINE_LOG_KEY") or ""
    if not url or not key or not (transcript or "").strip():
        return
    lang = {
        "English": "en",
        "Yoruba": "yo",
        "Hausa": "ha",
        "Igbo": "ig",
        "en": "en",
        "yo": "yo",
        "ha": "ha",
        "ig": "ig",
    }.get(language or "en", "en")
    try:
        import json
        import urllib.request

        req = urllib.request.Request(
            url if url.endswith("/interactions") else f"{url}/interactions",
            data=json.dumps(
                {
                    "question_text": transcript,
                    "response_text": reply,
                    "language": lang,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": key,
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=15)
    except Exception:
        return


def answer(audio_path: str | None, language: str) -> tuple[str, str, str | None]:
    """Plain Gradio handler — GPU decorator must not wrap this (breaks /info)."""
    if not audio_path:
        return "", "Record or upload a short voice note first.", None
    transcript, reply, audio_out = _run_on_gpu(audio_path, language)
    _log_to_hotline(transcript, reply, language)
    return transcript, reply, audio_out


with gr.Blocks(title="N-ATLAS Hotline") as demo:
    gr.Markdown(
        """
# N-ATLAS Hotline
Speak in **English, Yoruba, Hausa, or Igbo**. Hear and answer with official [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) weights.

**N-ATLaS does not speak.** Playback is a separate renderer: [Soro-TTS](https://huggingface.co/Shinzmann/soro-tts-yor) (MMS/VITS) for Yoruba, Igbo, and Hausa; gTTS for English.

First request can take 1–2 minutes (GPU + model load). Use a **short** voice note.
        """
    )
    with gr.Row():
        audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Voice note",
        )
        language = gr.Radio(
            choices=["English", "Yoruba", "Hausa", "Igbo"],
            value="English",
            label="Language",
        )
    btn = gr.Button("Ask N-ATLaS", variant="primary")
    transcript = gr.Textbox(label="What we heard (NCAIR ASR)")
    reply = gr.Textbox(label="N-ATLaS reply")
    spoken = gr.Audio(
        label="Spoken reply (not N-ATLaS — Soro-TTS / gTTS)",
        type="filepath",
    )
    btn.click(
        answer,
        inputs=[audio, language],
        outputs=[transcript, reply, spoken],
        api_name=False,
    )
    gr.Markdown(
        "N-ATLaS is an initiative of the Federal Ministry of Communications, "
        "Innovation and Digital Economy, powered by Awarri Technologies."
    )

demo.queue()

if __name__ == "__main__":
    demo.launch(ssr_mode=False, show_api=False)
