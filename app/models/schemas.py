from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class Language(str, Enum):
    YORUBA = "yo"
    HAUSA = "ha"
    IGBO = "ig"
    ENGLISH = "en"


class Channel(str, Enum):
    PHONE = "phone"
    WHATSAPP = "whatsapp"
    SPACE = "space"


class Domain(str, Enum):
    GOVERNMENT = "government"
    HEALTH = "health"
    MARKET = "market"
    AGRICULTURE = "agriculture"
    GENERAL = "general"


class RetrievalResult(BaseModel):
    chunks: list[str] = []
    sources: list[str] = []
    top_score: float = 0.0
    is_out_of_knowledge: bool = True


class PipelineRequest(BaseModel):
    session_id: str
    channel: Channel
    audio_path: str
    language: Language
    phone_number: str


class PipelineResult(BaseModel):
    session_id: str
    question_text: str
    intent_domain: Domain
    response_text: str
    response_audio_path: str
    rag_confidence: float
    out_of_knowledge: bool
    processing_time_ms: int
    retrieved_sources: list[str]


class InteractionRecord(BaseModel):
    session_id: str
    channel: Channel
    user_phone_hash: str
    language: Language
    question_text: str
    intent_domain: Domain
    response_text: str
    rag_confidence: float
    out_of_knowledge: bool
    processing_time_ms: int
    retrieved_sources: list[str]


class MarketPrice(BaseModel):
    commodity: str
    market_name: str
    state: str
    price: float
    unit: str
    source: str = "crowd"
    reported_at: datetime | None = None


class StatsResponse(BaseModel):
    total_interactions: int
    by_channel: dict[str, int]
    by_language: dict[str, int]
    by_domain: dict[str, int]
    out_of_knowledge_count: int
    avg_response_time_ms: float


class SpaceLogIn(BaseModel):
    """Q&A from the Hugging Face Space — no phone, no audio."""

    question_text: str
    response_text: str
    language: Language
    session_id: str = ""
