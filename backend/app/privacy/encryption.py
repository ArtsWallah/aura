"""Authenticated encryption for uploads and local task artifacts.

Fernet uses AES-128-CBC with HMAC-SHA256 authentication. The key must be
provided as a URL-safe base64 Fernet key via ENCRYPTION_KEY.
"""
from pathlib import Path
import os
import uuid

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings


class AtRestEncryption:
    def __init__(self, key: str | bytes | None = None):
        configured = key if key is not None else settings.ENCRYPTION_KEY
        if not configured:
            raise RuntimeError("ENCRYPTION_KEY is required for encrypted storage.")
        try:
            self.cipher = Fernet(configured.encode() if isinstance(configured, str) else configured)
        except (TypeError, ValueError) as exc:
            raise ValueError("ENCRYPTION_KEY must be a valid URL-safe base64 Fernet key.") from exc

    def encrypt_bytes(self, raw_data: bytes) -> bytes:
        return self.cipher.encrypt(raw_data)

    def decrypt_bytes(self, encrypted_data: bytes) -> bytes:
        try:
            return self.cipher.decrypt(encrypted_data)
        except InvalidToken as exc:
            raise ValueError("Encrypted data could not be authenticated; check ENCRYPTION_KEY.") from exc

    def encrypt_file(self, file_path: str | Path, out_path: str | Path | None = None) -> Path:
        src = Path(file_path)
        dest = Path(out_path) if out_path else src.with_suffix(src.suffix + ".enc")
        dest.write_bytes(self.encrypt_bytes(src.read_bytes()))
        return dest

    def decrypt_file(self, enc_file_path: str | Path, out_path: str | Path | None = None) -> Path:
        src = Path(enc_file_path)
        dest = Path(out_path) if out_path else src.with_suffix("")
        dest.write_bytes(self.decrypt_bytes(src.read_bytes()))
        return dest

    def encrypt_file_in_place(self, path: str | Path) -> None:
        """Atomically replace a legacy plaintext file with ciphertext at the same path."""
        target = Path(path)
        temporary: Path | None = None
        temporary_identity: tuple[int, int] | None = None
        descriptor: int | None = None
        try:
            # Exclusive creation prevents a collision from truncating an
            # unrelated file. Retry names are in the target directory so the
            # final replace remains atomic on the same filesystem.
            for _ in range(10):
                candidate = target.with_name(
                    f".{target.name}.{uuid.uuid4().hex}.encrypting"
                )
                try:
                    descriptor = os.open(
                        candidate,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
                        0o600,
                    )
                    temporary = candidate
                    info = os.fstat(descriptor)
                    temporary_identity = (info.st_dev, info.st_ino)
                    break
                except FileExistsError:
                    continue
            if descriptor is None or temporary is None:
                raise FileExistsError("Could not create a unique encryption staging file.")

            raw = target.read_bytes()
            encrypted = self.encrypt_bytes(raw)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(encrypted)
                stream.flush()
                os.fsync(stream.fileno())
            if self.decrypt_bytes(temporary.read_bytes()) != raw:
                raise ValueError("Encrypted file verification failed.")
            os.replace(temporary, target)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None and temporary_identity is not None:
                try:
                    current = temporary.lstat()
                    if (current.st_dev, current.st_ino) == temporary_identity:
                        temporary.unlink()
                except FileNotFoundError:
                    pass


def encryption_service() -> AtRestEncryption:
    """Resolve the current configured key lazily so missing keys fail closed."""
    return AtRestEncryption()


# Compatibility name; construction is intentionally deferred until first use.
class LazyEncryption:
    def encrypt_bytes(self, data: bytes) -> bytes:
        return encryption_service().encrypt_bytes(data)

    def decrypt_bytes(self, data: bytes) -> bytes:
        return encryption_service().decrypt_bytes(data)

    def encrypt_file(self, *args, **kwargs):
        return encryption_service().encrypt_file(*args, **kwargs)

    def decrypt_file(self, *args, **kwargs):
        return encryption_service().decrypt_file(*args, **kwargs)


at_rest_encryption = LazyEncryption()


def migrate_plaintext_files(paths: list[Path]) -> list[str]:
    """Encrypt legacy files in place; filenames and bytes after decryption are preserved."""
    cipher = encryption_service()
    migrated: list[str] = []
    for path in paths:
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.read_bytes().startswith(b"gAAAA"):
            continue
        cipher.encrypt_file_in_place(path)
        migrated.append(path.name)
    return migrated
