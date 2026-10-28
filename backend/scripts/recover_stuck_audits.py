
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

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
STALE_AFTER_MINUTES = 15


def get_active_job_audit_ids() -> set[int]:
    """Collect audit IDs from RQ's active registries."""
    audit_queue.connection.ping()

    job_ids = set(audit_queue.job_ids)

    registries = (
        StartedJobRegistry(queue=audit_queue),
        DeferredJobRegistry(queue=audit_queue),
        ScheduledJobRegistry(queue=audit_queue),
    )

    for registry in registries:
        job_ids.update(registry.get_job_ids())

    audit_ids: set[int] = set()

    for job_id in job_ids:
        try:
            job = Job.fetch(
                job_id,
                connection=audit_queue.connection,
            )
            args = job.args or ()

            if len(args) >= 2 and isinstance(args[0], int):
                audit_ids.add(args[0])

        except Exception as exc:
            raise RuntimeError(
                f"Could not inspect RQ job {job_id}; "
                "aborting recovery scan."
            ) from exc

    return audit_ids


def as_utc(value: datetime) -> datetime:
    """Interpret naive SQLite timestamps as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def main() -> None:
    print("MedShield-AI stuck-audit recovery scan (DRY RUN)")
    print(f"Stale threshold: {STALE_AFTER_MINUTES} minutes")

    active_job_audit_ids = get_active_job_audit_ids()
    cutoff = datetime.now(timezone.utc) - timedelta(
        minutes=STALE_AFTER_MINUTES
    )

    staging_dir = Path(settings.STAGING_DIR).resolve()
    db = SessionLocal()

    try:
        audits = (
            db.query(AuditSession)
            .filter(AuditSession.status.in_(ACTIVE_STATUSES))
            .order_by(AuditSession.id)
            .all()
        )

        if not audits:
            print("No queued or processing audits found.")
            return

        for audit in audits:
            updated_at = audit.updated_at or audit.created_at
            age = datetime.now(timezone.utc) - as_utc(updated_at)

            if audit.id in active_job_audit_ids:
                print(
                    f"KEEP audit_id={audit.id}: "
                    "active RQ job found."
                )
                continue

            if as_utc(updated_at) > cutoff:
                print(
                    f"WAIT audit_id={audit.id}: "
                    f"age={age.total_seconds() / 60:.1f} minutes; "
                    "below stale threshold."
                )
                continue

            if not audit.storage_id:
                print(
                    f"ALERT audit_id={audit.id}: "
                    "stale record has no storage_id. Manual review required."
                )
                continue

            staging_file = (
                staging_dir / f"{audit.storage_id}.enc"
            )

            if not staging_file.is_file():
                print(
                    f"ALERT audit_id={audit.id}: "
                    "staging file is missing. Manual review required."
                )
                continue

            print(
                f"REVIEW audit_id={audit.id}: "
                f"status={audit.status}; "
                f"age={age.total_seconds() / 60:.1f} minutes; "
                "no active RQ job found; "
                "staging file exists. No changes made."
            )

        print("DRY RUN complete. No database records or RQ jobs were changed.")

    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"Recovery scan aborted safely: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
