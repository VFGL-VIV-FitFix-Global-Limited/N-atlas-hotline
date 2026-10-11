from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class TempFileManager:
    """Track and clean up temporary audio files."""

    def __init__(self) -> None:
        self._files: list[str] = []

    def register(self, path: str) -> str:
        self._files.append(path)
        return path

    def cleanup(self) -> None:
        for f in self._files:
            try:
                Path(f).unlink(missing_ok=True)
            except OSError:
                logger.warning("cleanup_failed", path=f)
        self._files.clear()

    def __enter__(self) -> "TempFileManager":
        return self

    def __exit__(self, *args: object) -> None:
        self.cleanup()
