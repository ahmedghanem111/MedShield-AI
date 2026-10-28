
import os
import re
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import settings
from app.services.encrypted_temp import _encryption_key

_MAGIC = b"MSS1"
_NONCE_SIZE = 12


def _staging_directory() -> Path:
    directory = Path(settings.STAGING_DIR).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    if os.name != "nt":
        os.chmod(directory, 0o700)

    return directory


def store_encrypted_upload(audit_id: int, content: bytes) -> str:
    if audit_id < 1:
        raise ValueError("Invalid audit ID")

    directory = _staging_directory()
    storage_id = uuid4().hex
    nonce = os.urandom(_NONCE_SIZE)
    cipher = AESGCM(_encryption_key())

    associated_data = f"medshield:staging:v1:{audit_id}".encode()
    ciphertext = cipher.encrypt(nonce, content, associated_data)
    payload = _MAGIC + nonce + ciphertext

    path = directory / f"{storage_id}.enc"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".pending-",
        dir=directory,
    )

    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary_name, path)
        os.utime(path, (time.time(), time.time()))
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    return storage_id


def read_encrypted_upload(audit_id: int, storage_id: str) -> bytes:
    if audit_id < 1 or not re.fullmatch(r"[0-9a-f]{32}", storage_id):
        raise ValueError("Invalid staging reference")

    path = _staging_directory() / f"{storage_id}.enc"
    payload = path.read_bytes()

    if len(payload) < len(_MAGIC) + _NONCE_SIZE + 16:
        raise ValueError("Invalid encrypted staging file")

    if not payload.startswith(_MAGIC):
        raise ValueError("Invalid encrypted staging file")

    nonce_start = len(_MAGIC)
    nonce = payload[nonce_start:nonce_start + _NONCE_SIZE]
    ciphertext = payload[nonce_start + _NONCE_SIZE:]

    associated_data = f"medshield:staging:v1:{audit_id}".encode()
    return AESGCM(_encryption_key()).decrypt(
        nonce,
        ciphertext,
        associated_data,
    )


def delete_staged_upload(storage_id: str) -> bool:
    if not re.fullmatch(r"[0-9a-f]{32}", storage_id):
        raise ValueError("Invalid staging reference")

    path = _staging_directory() / f"{storage_id}.enc"

    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
