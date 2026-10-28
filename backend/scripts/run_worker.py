
import sys
from pathlib import Path

# Add the backend root directory to Python's import path.
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from rq import SimpleWorker

from app.core.queue import audit_queue, redis_connection


if __name__ == "__main__":
    worker = SimpleWorker(
        [audit_queue],
        connection=redis_connection,
    )

    print("MedShield-AI SimpleWorker is starting...", flush=True)
    worker.work(with_scheduler=False)
