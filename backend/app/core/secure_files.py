"""Encrypted, request-scoped temporary file storage."""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterator

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from app.core.config import settings

_MAGIC = b"MSF1"
_NONCE_SIZE = 12


def _encryption_key() -> bytes:
    configured_key = settings.TEMP_FILE_ENCRYPTION_KEY or settings.SECRET_KEY
    if not configured_key:
        raise RuntimeError("A secret key is required for temporary file encryption")
    return HKDF(
        algorithm=SHA256(), length=32, salt=None,
        info=b"medshield/temporary-files/v1",
    ).derive(configured_key.encode("utf-8"))


class EncryptedTemporaryFile:
    """An encrypted temporary file with authenticated reads and writes."""

    def __init__(self, directory: Path):
        descriptor, filename = tempfile.mkstemp(prefix="medshield-", suffix=".enc", dir=directory)
        os.close(descriptor)
        self.path = Path(filename)
        self._cipher = AESGCM(_encryption_key())

    def write(self, plaintext: bytes) -> None:
        nonce = os.urandom(_NONCE_SIZE)
        ciphertext = self._cipher.encrypt(nonce, plaintext, _MAGIC)
        self.path.write_bytes(_MAGIC + nonce + ciphertext)

    def read(self) -> bytes:
        payload = self.path.read_bytes()
        if len(payload) < len(_MAGIC) + _NONCE_SIZE + 16 or not payload.startswith(_MAGIC):
            raise ValueError("Invalid encrypted temporary file")
        nonce_start = len(_MAGIC)
        nonce = payload[nonce_start : nonce_start + _NONCE_SIZE]
        return self._cipher.decrypt(nonce, payload[nonce_start + _NONCE_SIZE :], _MAGIC)


@contextmanager
def encrypted_temporary_file(
    directory: str | os.PathLike[str] | None = None,
) -> Iterator[EncryptedTemporaryFile]:
    """Create a private temporary file and guarantee its removal on exit."""
    with tempfile.TemporaryDirectory(prefix="medshield-tmp-", dir=directory) as temp_dir:
        os.chmod(temp_dir, 0o700)
        encrypted_file = EncryptedTemporaryFile(Path(temp_dir))
        try:
            yield encrypted_file
        finally:
            # Best effort only: filesystem wear leveling prevents guaranteed erasure.
            try:
                size = encrypted_file.path.stat().st_size
                with encrypted_file.path.open("r+b", buffering=0) as stream:
                    stream.write(os.urandom(size))
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError:
                pass
