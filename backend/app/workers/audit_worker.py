import logging
from sqlalchemy.orm import Session
from app.db import base  # Register all SQLAlchemy models
from app.db.database import SessionLocal
from app.models.audit_model import AuditSession
from datetime import datetime, timezone
from app.services.staging_storage import (
    delete_staged_upload,
    read_encrypted_upload,
)
from app.services.zip_service import validate_and_read_zip

logger = logging.getLogger(__name__)


def process_audit(audit_id: int, storage_id: str) -> dict:
    db: Session = SessionLocal()
    owns_processing = False

    try:
        audit = (
            db.query(AuditSession)
            .filter(AuditSession.id == audit_id)
            .first()
        )

        if audit is None:
            raise ValueError("Audit session not found")

        if audit.status != "queued":
            return {
                "audit_id": audit_id,
                "status": audit.status,
                "message": "Audit is not queued",
            }

        # Claim the audit before processing it.
        audit.status = "processing"
        audit.updated_at = datetime.now(timezone.utc)
        db.commit()
        owns_processing = True

        encrypted_zip = read_encrypted_upload(audit_id, storage_id)
        files = validate_and_read_zip(encrypted_zip)

        if len(files) != audit.files_count:
            raise ValueError(
                "ZIP file count does not match audit record"
            )

        # ZIP validation succeeded. AI inference will be added later.
        audit.status = "validated"
        audit.updated_at = datetime.now(timezone.utc)
        db.commit()
        owns_processing = False

        # File cleanup is separate from validation success.
        # A cleanup error must not turn a validated audit into failed.
        try:
            deleted = delete_staged_upload(storage_id)

            if not deleted:
                logger.warning(
                    "Staged file was already missing: audit_id=%s",
                    audit_id,
                )

        except Exception:
            logger.exception(
                "Could not delete staged file after validation: "
                "audit_id=%s",
                audit_id,
            )

        return {
            "audit_id": audit_id,
            "status": "validated",
            "files_count": len(files),
        }

    except Exception:
        db.rollback()

        if owns_processing:
            try:
                audit = (
                    db.query(AuditSession)
                    .filter(AuditSession.id == audit_id)
                    .first()
                )

                if audit is not None and audit.status == "processing":
                    audit.status = "failed"
                    audit.updated_at = datetime.now(timezone.utc)
                    db.commit()

            except Exception:
                db.rollback()
                logger.exception(
                    "Could not mark audit as failed: audit_id=%s",
                    audit_id,
                )

        logger.exception("Audit processing failed: audit_id=%s", audit_id)
        raise

    finally:
        db.close()
