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

**Demo default for WhatsApp is `NATLAS_BACKEND=space`:** Railway calls the Hugging Face Space (`HF_SPACE_ID`, default `onomeakpobaro/hotline`) so evaluators and users share one N-ATLaS GPU. The Space loads weights with `from_pretrained` as on the model card.

`gateway` is the Colab / OpenAI-compatible path (`NATLAS_BASE_URL` → `/v1/audio/transcriptions` and `/v1/chat/completions`). Fragile on trycloudflare (HTTP 524 if the 8B is slow or CPU-offloaded).

`local` is a GPU workstation only. Do not use it on Railway CPU. `huggingface` serverless does not host NCAIR1; dedicated Inference Endpoints only.

## Attribution

> N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, powered by Awarri Technologies.
