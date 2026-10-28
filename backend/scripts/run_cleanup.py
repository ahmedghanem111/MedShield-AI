import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.staging_cleanup import cleanup_stale_staging_files


if __name__ == "__main__":
    print("MedShield-AI staging cleanup started.", flush=True)

    try:
        result = cleanup_stale_staging_files()
        print(f"Cleanup result: {result}", flush=True)
    except Exception as exc:
        print(
            f"Staging cleanup failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
