def normalize_ng_phone(number: str) -> str:
    """Normalise Nigerian numbers to E.164 (+234...)."""
    raw = number.strip().replace(" ", "").replace("-", "")
    if raw.startswith("00"):
        raw = "+" + raw[2:]
    if raw.startswith("0") and len(raw) == 11:
        return "+234" + raw[1:]
    if raw.startswith("234") and not raw.startswith("+"):
        return "+" + raw
    return raw


def is_nigerian_number(number: str) -> bool:
    return normalize_ng_phone(number).startswith("+234")
