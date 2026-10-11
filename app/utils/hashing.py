import hashlib


def hash_phone(phone_number: str, salt: str) -> str:
    """Hash a phone number with salt using SHA-256. Irreversible."""
    return hashlib.sha256(f"{salt}:{phone_number}".encode()).hexdigest()
