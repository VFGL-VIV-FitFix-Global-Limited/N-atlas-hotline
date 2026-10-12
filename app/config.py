from enum import Enum

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


def natlas_origin(base: str) -> str:
    """Gateway origin (health lives on the host root, not under /v1)."""
    root = base.rstrip("/")
    if root.endswith("/v1"):
        return root[:-3].rstrip("/")
    return root


def natlas_v1_url(base: str, path: str) -> str:
    """Join NATLAS_BASE_URL (with or without /v1) to a /v1 path."""
    root = base.rstrip("/")
    if root.endswith("/v1"):
        return f"{root}/{path.lstrip('/')}"
    return f"{root}/v1/{path.lstrip('/')}"


class Settings(BaseSettings):
    # gateway (default) = hosted HTTP API — use this on Railway/Render
    # huggingface = HF Inference API
    # local = from_pretrained (GPU only; not for production)
    natlas_backend: str = "gateway"
    natlas_device: str = "auto"
    hf_token: str = ""
    natlas_base_url: str = ""
    natlas_asr_url: str = "https://api.natlas.ng/asr"
    natlas_llm_url: str = "https://api.natlas.ng/llm"
    natlas_api_key: str = ""
    natlas_model: str = Field(
        default="NCAIR1/N-ATLaS",
        validation_alias=AliasChoices("natlas_model", "hf_model_id"),
    )

    # Africa's Talking
    at_username: str = ""
    at_api_key: str = ""
    at_phone_number: str = ""

    # WhatsApp Cloud API
    wa_phone_number_id: str = ""
    wa_access_token: str = ""
    wa_verify_token: str = ""
    wa_app_secret: str = ""  # HMAC for X-Hub-Signature-256; empty = skip (dev only)
    wa_graph_version: str = "v21.0"

    # Twilio WhatsApp Sandbox
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = "+14155238886"

    # App
    app_base_url: str = "http://localhost:8000"
    database_path: str = "data/hotline.db"
    knowledge_path: str = "knowledge/"
    faiss_index_path: str = "data/faiss_index"
    audio_dir: str = "data/audio"
    log_level: str = "INFO"
    environment: Environment = Environment.DEVELOPMENT

    # Security
    api_secret_key: str = "change-me-in-production"
    phone_hash_salt: str = "dev-salt-16chars"

    # Rate limiting
    rate_limit_calls_per_minute: int = 10
    rate_limit_whatsapp_per_minute: int = 20

    # Pipeline — N-ATLAS demo gateway cold-starts; 60s matches SDK default
    rag_confidence_threshold: float = 0.35
    max_response_words: int = 150
    max_call_duration_seconds: int = 180
    max_recording_seconds: int = 30
    asr_timeout_seconds: int = 10
    llm_timeout_seconds: int = 15

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "populate_by_name": True}

    @property
    def natlas_health_url(self) -> str:
        base = self.natlas_base_url or self.natlas_llm_url
        return f"{natlas_origin(base)}/health"

    @property
    def wa_graph_base(self) -> str:
        return f"https://graph.facebook.com/{self.wa_graph_version}"

    @field_validator("natlas_backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = {"local", "huggingface", "gateway"}
        if v not in allowed:
            raise ValueError(f"natlas_backend must be one of {allowed}")
        return v

    @field_validator("natlas_base_url", "natlas_asr_url", "natlas_llm_url", "app_base_url")
    @classmethod
    def validate_urls(cls, v: str) -> str:
        if not v:
            return v
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {v} — must start with http:// or https://")
        return v.rstrip("/")

    @model_validator(mode="after")
    def fill_gateway_urls_from_base(self) -> "Settings":
        """If NATLAS_BASE_URL is set, derive OpenAI-compatible /v1 paths unless URLs were customised."""
        if self.natlas_base_url:
            defaults = {"https://api.natlas.ng/asr", "https://api.natlas.ng/llm"}
            if self.natlas_asr_url in defaults:
                self.natlas_asr_url = natlas_v1_url(self.natlas_base_url, "audio/transcriptions")
            if self.natlas_llm_url in defaults:
                self.natlas_llm_url = natlas_v1_url(self.natlas_base_url, "chat/completions")
        return self

    @field_validator("phone_hash_salt")
    @classmethod
    def validate_salt_length(cls, v: str) -> str:
        if len(v) < 16:
            raise ValueError("phone_hash_salt must be at least 16 characters")
        return v

    @field_validator("natlas_model")
    @classmethod
    def validate_natlas_model(cls, v: str) -> str:
        if not v.startswith("NCAIR1/"):
            raise ValueError("natlas_model must start with NCAIR1/")
        return v
