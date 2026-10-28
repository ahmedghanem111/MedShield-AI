
from datetime import timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.queue import audit_queue
from app.core.security import get_current_user_id
from app.db.database import get_db
from app.models.audit_model import AuditSession
from app.services.staging_storage import (
    store_encrypted_upload,
    delete_staged_upload,
)
from app.services.zip_service import (
    MAX_ARCHIVE_SIZE,
    ZipValidationError,
    validate_and_read_zip,
)
from app.workers.audit_worker import process_audit


router = APIRouter(prefix="/api/uploads", tags=["Uploads"])


@router.post("/zip", status_code=status.HTTP_202_ACCEPTED)
async def upload_zip(
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a ZIP file",
        )

    try:
        zip_bytes = await file.read(MAX_ARCHIVE_SIZE + 1)
    finally:
        await file.close()

    if len(zip_bytes) > MAX_ARCHIVE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="ZIP archive exceeds the 25 MB limit",
        )

    try:
        files = validate_and_read_zip(zip_bytes)
    except ZipValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    audit = AuditSession(
        user_id=user_id,
        original_filename=file.filename[:255],
        status="queued",
        files_count=len(files),
    )

    db.add(audit)

    try:
        db.commit()
        db.refresh(audit)
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create audit session",
        )

    storage_id = None

    try:
        storage_id = store_encrypted_upload(audit.id, zip_bytes)

        audit.storage_id = storage_id
        db.commit()
        db.refresh(audit)

        audit_queue.enqueue(
            process_audit,
            audit.id,
            storage_id,
            job_timeout=600,
            result_ttl=3600,
            failure_ttl=86400,
        )


    except Exception as exc:
        db.rollback()

        try:
            failed_audit = (
                db.query(AuditSession)
                .filter(AuditSession.id == audit.id)
                .first()
            )

            # Only mark it failed if the worker hasn't claimed it.
            if failed_audit is not None and failed_audit.status == "queued":
                failed_audit.status = "failed"
                db.commit()

        except Exception:
            db.rollback()

        # Do not delete the staged file here.
        # Enqueue may have succeeded even if the client saw an error.
        # Let the staging cleanup remove expired, inactive files safely.

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not confirm that the audit was queued. Check its status before retrying.",
        ) from exc


        if storage_id is not None:
            try:
                delete_staged_upload(storage_id)
            except (OSError, ValueError):
                pass

        try:
            failed_audit = (
                db.query(AuditSession)
                .filter(AuditSession.id == audit.id)
                .first()
            )

            if failed_audit is not None:
                failed_audit.status = "failed"
                db.commit()
        except Exception:
            db.rollback()

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not queue audit for processing",
        ) from exc

    return {
        "message": "ZIP accepted for background processing",
        "audit_id": audit.id,
        "status": "queued",
        "files_count": len(files),
        "files": [
            {
                "filename": item["filename"],
                "size": item["size"],
            }
            for item in files
        ],
    }


@router.get("/audits/{audit_id}")
def get_audit_session(
    audit_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    audit = (
        db.query(AuditSession)
        .filter(
            AuditSession.id == audit_id,
            AuditSession.user_id == user_id,
        )
        .first()
    )

    if audit is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audit session not found",
        )

    created_at = audit.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    return {
        "audit_id": audit.id,
        "status": audit.status,
        "files_count": audit.files_count,
        "original_filename": audit.original_filename,
        "created_at": created_at.isoformat(),
    }
