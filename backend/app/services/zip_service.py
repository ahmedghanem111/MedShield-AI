
"""Validate ZIP uploads without extracting files to disk."""

from io import BytesIO
from pathlib import PurePosixPath
import re
import stat
import zipfile

MAX_ARCHIVE_SIZE = 25 * 1024 * 1024  # 25 MB
MAX_FILES = 100
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB per file
MAX_TOTAL_SIZE = 50 * 1024 * 1024  # 50 MB uncompressed
MAX_COMPRESSION_RATIO = 100

ALLOWED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".heic",
    ".heif",
    ".dcm",
    ".pdf",
}


class ZipValidationError(ValueError):
    """Raised when a ZIP archive violates upload security rules."""


def _validate_member_name(name: str) -> str:
    # ZIP paths normally use forward slashes; reject Windows separators too.
    if not name or "\\" in name or "\x00" in name:
        raise ZipValidationError("Invalid file path in ZIP archive")

    if name.startswith("/") or re.match(r"^[a-zA-Z]:", name):
        raise ZipValidationError("Absolute paths are not allowed")

    path = PurePosixPath(name)

    if any(part in {".", ".."} for part in path.parts):
        raise ZipValidationError("Path traversal detected in ZIP archive")

    return name


def validate_and_read_zip(zip_bytes: bytes) -> list[dict]:
    """
    Validate a ZIP archive and return its allowed files in memory.

    No archive members are extracted to the filesystem.
    """
    if not zip_bytes:
        raise ZipValidationError("The uploaded ZIP file is empty")

    if len(zip_bytes) > MAX_ARCHIVE_SIZE:
        raise ZipValidationError("ZIP archive exceeds the 25 MB limit")

    if not zipfile.is_zipfile(BytesIO(zip_bytes)):
        raise ZipValidationError("Uploaded file is not a valid ZIP archive")

    extracted_files = []
    total_size = 0

    try:
        with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
            members = archive.infolist()

            files = [member for member in members if not member.is_dir()]

            if len(files) > MAX_FILES:
                raise ZipValidationError("ZIP archive contains too many files")

            for member in members:
                _validate_member_name(member.filename)

                if member.is_dir():
                    continue

                # Reject symbolic links and other non-regular Unix file types.
                unix_mode = member.external_attr >> 16
                file_type = stat.S_IFMT(unix_mode)

                if file_type not in {0, stat.S_IFREG}:
                    raise ZipValidationError(
                        "Links and special files are not allowed"
                    )

                if member.flag_bits & 0x1:
                    raise ZipValidationError(
                        "Password-protected ZIP entries are not supported"
                    )

                extension = PurePosixPath(member.filename).suffix.lower()

                if extension not in ALLOWED_EXTENSIONS:
                    raise ZipValidationError(
                        f"Unsupported file type: {extension or 'no extension'}"
                    )

                if member.file_size > MAX_FILE_SIZE:
                    raise ZipValidationError(
                        "A file exceeds the 20 MB per-file limit"
                    )

                total_size += member.file_size

                if total_size > MAX_TOTAL_SIZE:
                    raise ZipValidationError(
                        "Uncompressed ZIP contents exceed 50 MB"
                    )

                if member.file_size > 0:
                    if member.compress_size == 0:
                        raise ZipValidationError(
                            "Invalid ZIP compression metadata"
                        )

                    ratio = member.file_size / member.compress_size

                    if ratio > MAX_COMPRESSION_RATIO:
                        raise ZipValidationError(
                            "Suspicious compression ratio detected"
                        )

                # Read each file in memory; never extract its path to disk.
                with archive.open(member, "r") as source:
                    content = source.read(MAX_FILE_SIZE + 1)

                if len(content) != member.file_size:
                    raise ZipValidationError(
                        "File size does not match ZIP metadata"
                    )

                extracted_files.append({
                    "filename": PurePosixPath(member.filename).name,
                    "size": len(content),
                    "content": content,
                })

    except ZipValidationError:
        raise
    except (zipfile.BadZipFile, RuntimeError, OSError, EOFError) as exc:
        raise ZipValidationError(
            "The ZIP archive is damaged or cannot be read"
        ) from exc

    if not extracted_files:
        raise ZipValidationError("ZIP archive contains no supported files")

    return extracted_files
