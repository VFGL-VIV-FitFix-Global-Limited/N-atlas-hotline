"""Railway/Docker entrypoint. Reads PORT in Python — no shell glob, no $PORT."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    print(f"n-atlas-hotline binding 0.0.0.0:{port}", flush=True)
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_level="info",
    )


if __name__ == "__main__":
    main()
