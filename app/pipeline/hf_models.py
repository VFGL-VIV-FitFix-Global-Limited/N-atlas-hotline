"""Official N-ATLaS weights on Hugging Face.

LLM: https://huggingface.co/NCAIR1/N-ATLaS
ASR: Whisper Small checkpoints under NCAIR1, one language each.
"""

LLM_MODEL_ID = "NCAIR1/N-ATLaS"
LLM_HF_URL = "https://huggingface.co/NCAIR1/N-ATLaS"

# Model card generation settings
LLM_TEMPERATURE = 0.1
LLM_REPETITION_PENALTY = 1.12
LLM_MAX_NEW_TOKENS = 300

# Official NCAIR Whisper-small finetunes (same N-ATLaS programme).
ASR_MODEL_IDS: dict[str, str] = {
    "ha": "NCAIR1/Hausa-ASR",
    "ig": "NCAIR1/Igbo-ASR",
    "yo": "NCAIR1/Yoruba-ASR",
    "en": "NCAIR1/NigerianAccentedEnglish",
}

# Hugging Face serverless hosts these. NCAIR1 weights are not on hf-inference.
ASR_SERVERLESS_MODEL = "openai/whisper-large-v3"
LLM_SERVERLESS_MODEL = "Qwen/Qwen2.5-7B-Instruct"

ATTRIBUTION = (
    "N-ATLaS is an initiative of the Federal Ministry of Communications, "
    "Innovation and Digital Economy, powered by Awarri Technologies."
)
