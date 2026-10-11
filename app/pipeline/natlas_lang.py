"""Language codes accepted by N-ATLAS POST /v1/audio/transcriptions.

Unrecognized ASR language is HTTP 400. Chat `language` is optional and only logged.
"""

_ALIASES: dict[str, str] = {
    "ha": "ha",
    "hau": "ha",
    "hausa": "ha",
    "ha-ng": "ha",
    "ig": "ig",
    "ibo": "ig",
    "igbo": "ig",
    "ig-ng": "ig",
    "yo": "yo",
    "yor": "yo",
    "yoruba": "yo",
    "yorùbá": "yo",
    "yo-ng": "yo",
    "en": "en",
    "eng": "en",
    "english": "en",
    "en-ng": "en",
    "en-us": "en",
    "en-gb": "en",
    "nigerian english": "en",
    "naija": "en",
}


def normalise_natlas_language(language: str) -> str:
    key = language.strip().lower().replace("_", "-")
    if key not in _ALIASES:
        raise ValueError(f"Unsupported language {language!r}. Use ha, ig, yo, or en.")
    return _ALIASES[key]
