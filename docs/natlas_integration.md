# N-ATLAS Integration

Source of truth: **[NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS)**.

The LLM is loaded **in process**, the way the model card shows — not via a chat API:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "NCAIR1/N-ATLaS"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

Chat uses `tokenizer.apply_chat_template(..., date_string=today)` and `generate` with `temperature=0.1`, `repetition_penalty=1.12`.

ASR is the matching Whisper Small checkpoints, also local:

| Language | Repo |
|---|---|
| Hausa | `NCAIR1/Hausa-ASR` |
| Igbo | `NCAIR1/Igbo-ASR` |
| Yoruba | `NCAIR1/Yoruba-ASR` |
| Nigerian English | `NCAIR1/NigerianAccentedEnglish` |

**Default for this hotline is `NATLAS_BACKEND=gateway`.** That is the production path (Railway/Render): HTTP calls, no GPU on your server.

Set `NATLAS_ASR_URL`, `NATLAS_LLM_URL`, and `NATLAS_API_KEY`. If the host is OpenAI-compatible (vLLM / TGI / the N-ATLAS kit), you can set `NATLAS_BASE_URL` instead and we derive `/v1/audio/transcriptions` and `/v1/chat/completions`.

`local` is optional and only for a GPU workstation (`AutoModelForCausalLM.from_pretrained("NCAIR1/N-ATLaS")`). Do not use it on Railway. `huggingface` uses the HF Inference API with the same repo IDs.

## Attribution

> N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, powered by Awarri Technologies.
