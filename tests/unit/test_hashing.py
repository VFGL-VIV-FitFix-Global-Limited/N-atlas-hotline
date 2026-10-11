from app.utils.hashing import hash_phone


def test_same_input_same_hash() -> None:
    assert hash_phone("+2348012345678", "salt-sixteen-xx") == hash_phone("+2348012345678", "salt-sixteen-xx")


def test_different_inputs_different_hashes() -> None:
    a = hash_phone("+2348012345678", "salt-sixteen-xx")
    b = hash_phone("+2348099999999", "salt-sixteen-xx")
    assert a != b


def test_different_salts_different_hashes() -> None:
    a = hash_phone("+2348012345678", "salt-sixteen-aa")
    b = hash_phone("+2348012345678", "salt-sixteen-bb")
    assert a != b


def test_hash_is_sha256_hex() -> None:
    digest = hash_phone("+2348012345678", "salt-sixteen-xx")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
