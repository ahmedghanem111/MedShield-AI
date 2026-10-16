
from app.services.encrypted_temp import encrypted_temporary_file


secret_data = b"MedShield AI - confidential medical test data"

saved_path = None

with encrypted_temporary_file() as encrypted_file:
    saved_path = encrypted_file.path

    # Write encrypted data to disk
    encrypted_file.write(secret_data)

    stored_data = saved_path.read_bytes()

    # Verify plaintext is not stored directly
    assert stored_data != secret_data
    print("Encryption: PASS")

    # Verify decryption restores the original data
    decrypted_data = encrypted_file.read()

    assert decrypted_data == secret_data
    print("Decryption: PASS")

# Verify automatic cleanup after leaving the context
assert saved_path is not None
assert not saved_path.exists()

print("Temporary file cleanup: PASS")
print("All encryption tests passed!")