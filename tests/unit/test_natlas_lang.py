import pytest

from app.pipeline.natlas_lang import normalise_natlas_language


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ha", "ha"),
        ("Hausa", "ha"),
        ("ha_NG", "ha"),
        ("igbo", "ig"),
        ("Yorùbá", "yo"),
        ("en-NG", "en"),
        ("naija", "en"),
        ("Nigerian English", "en"),
    ],
)
def test_language_aliases(raw: str, expected: str) -> None:
    assert normalise_natlas_language(raw) == expected


def test_unsupported_language() -> None:
    with pytest.raises(ValueError, match="Unsupported language"):
        normalise_natlas_language("fr")
