
import re
import time
from pathlib import Path

from app.core.config import settings
from app.core.queue import audit_queue
from app.db import base
from app.db.database import SessionLocal
from app.models.audit_model import AuditSession
from rq.job import Job
from rq.registry import (
    DeferredJobRegistry,
    ScheduledJobRegistry,
    StartedJobRegistry,
)


ACTIVE_STATUSES = {"queued", "processing"}
STORAGE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _storage_id_from_job(job: Job) -> str | None:
    args = job.args or ()

    if len(args) < 2:
        return None

    storage_id = args[1]

    if not isinstance(storage_id, str):
        return None

    if not STORAGE_ID_PATTERN.fullmatch(storage_id):
        return None

    return storage_id


def _active_storage_ids() -> set[str]:
    """Return storage IDs referenced by queued or potentially running jobs."""
    connection = audit_queue.connection
    active_ids: set[str] = set()

    job_ids = set(audit_queue.job_ids)

    registries = [
        StartedJobRegistry(queue=audit_queue),
        DeferredJobRegistry(queue=audit_queue),
        ScheduledJobRegistry(queue=audit_queue),
    ]

    for registry in registries:
        job_ids.update(registry.get_job_ids())

    for job_id in job_ids:
        try:
            job = Job.fetch(job_id, connection=connection)
            storage_id = _storage_id_from_job(job)
            if storage_id:
                active_ids.add(storage_id)
        except Exception as exc:
            raise RuntimeError(
                f"Could not inspect RQ job {job_id}; "
                "aborting staging cleanup."
            ) from exc

    return active_ids


def cleanup_stale_staging_files() -> dict[str, int]:
    """Delete expired staging files not associated with active audits."""
    directory = Path(settings.STAGING_DIR).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    ttl_seconds = max(0, settings.STAGING_TTL_HOURS) * 3600
    cutoff = time.time() - ttl_seconds

    deleted = 0
    skipped_active = 0
    skipped_recent = 0
    errors = 0

    try:
        active_storage_ids = _active_storage_ids()
    except Exception:
        # If Redis is unavailable, fail closed: delete nothing.
        return {
            "deleted": 0,
            "skipped_active": 0,
            "skipped_recent": 0,
            "errors": 1,
        }

    db = SessionLocal()

    try:
        for path in directory.iterdir():
            try:
                if not path.is_file():
                    continue

                modified_at = path.stat().st_mtime

                if modified_at >= cutoff:
                    skipped_recent += 1
                    continue

                if path.name.startswith(".pending-"):
                    path.unlink()
                    deleted += 1
                    continue

                match = re.fullmatch(r"([0-9a-f]{32})\.enc", path.name)
                if not match:
                    continue

                storage_id = match.group(1)

                if storage_id in active_storage_ids:
                    skipped_active += 1
                    continue

                audit = (
                    db.query(AuditSession)
                    .filter(AuditSession.storage_id == storage_id)
                    .first()
                )

                if audit is not None and audit.status in ACTIVE_STATUSES:
                    skipped_active += 1
                    continue

                path.unlink()
                deleted += 1

            except OSError:
                errors += 1
            except Exception:
                errors += 1

    finally:
        db.close()

    return {
        "deleted": deleted,
        "skipped_active": skipped_active,
        "skipped_recent": skipped_recent,
        "errors": errors,
    }
