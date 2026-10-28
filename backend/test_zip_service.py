
from io import BytesIO
import zipfile

from app.services.zip_service import (
    validate_and_read_zip,
    ZipValidationError,
)


def make_zip(filename: str, content: bytes) -> bytes:
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(filename, content)

    return buffer.getvalue()


# Test 1: Valid ZIP
files = validate_and_read_zip(
    make_zip("scan.png", b"fake-image-content")
)
assert files[0]["filename"] == "scan.png"
assert files[0]["content"] == b"fake-image-content"
print("Valid ZIP: PASS")


# Test 2: Unsupported file extension
try:
    validate_and_read_zip(make_zip("script.exe", b"test"))
    raise AssertionError("Unsupported extension was accepted")
except ZipValidationError:
    print("Unsupported extension: PASS")


# Test 3: Path traversal
try:
    validate_and_read_zip(make_zip("../outside.png", b"test"))
    raise AssertionError("Path traversal was accepted")
except ZipValidationError:
    print("Path traversal protection: PASS")


# Test 4: Invalid ZIP
try:
    validate_and_read_zip(b"this is not a zip")
    raise AssertionError("Invalid ZIP was accepted")
except ZipValidationError:
    print("Invalid ZIP: PASS")


print("All ZIP validation tests passed!")
