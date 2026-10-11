from app.channels.whatsapp import verify_whatsapp_signature


def test_skip_when_secret_empty() -> None:
    assert verify_whatsapp_signature(b"{}", None, "") is True


def test_valid_signature() -> None:
    import hashlib
    import hmac

    secret = "app-secret"
    body = b'{"ok":true}'
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_whatsapp_signature(body, f"sha256={digest}", secret) is True


def test_invalid_signature() -> None:
    assert verify_whatsapp_signature(b"{}", "sha256=deadbeef", "app-secret") is False
