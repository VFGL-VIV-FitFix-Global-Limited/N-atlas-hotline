from pathlib import Path

import pytest

from app.utils.audio import convert_to_wav


async def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        await convert_to_wav("/tmp/does-not-exist.wav")


async def test_unsupported_format_raises(tmp_path: Path) -> None:
    bad = tmp_path / "note.txt"
    bad.write_text("not audio")
    with pytest.raises(ValueError):
        await convert_to_wav(str(bad))
