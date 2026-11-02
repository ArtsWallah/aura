import base64
import os
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from app.core.config import settings
from app.core.logging import logger

class AtRestEncryption:
    def __init__(self, secret: str | None = None):
        secret_bytes = (secret or settings.ENCRYPTION_KEY).encode("utf-8")
        salt = b"aura_secure_salt_2026"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100_000
        )
        key = base64.urlsafe_b64encode(kdf.derive(secret_bytes))
        self.cipher = Fernet(key)

    def encrypt_bytes(self, raw_data: bytes) -> bytes:
        return self.cipher.encrypt(raw_data)

    def decrypt_bytes(self, encrypted_data: bytes) -> bytes:
        return self.cipher.decrypt(encrypted_data)

    def encrypt_file(self, file_path: str | Path, out_path: str | Path | None = None) -> Path:
        src = Path(file_path)
        dest = Path(out_path) if out_path else src.with_suffix(src.suffix + ".enc")
        
        with open(src, "rb") as f:
            data = f.read()
        encrypted = self.encrypt_bytes(data)
        with open(dest, "wb") as f:
            f.write(encrypted)
            
        return dest

    def decrypt_file(self, enc_file_path: str | Path, out_path: str | Path | None = None) -> Path:
        src = Path(enc_file_path)
        dest = Path(out_path) if out_path else src.with_suffix("")
        
        with open(src, "rb") as f:
            enc_data = f.read()
        decrypted = self.decrypt_bytes(enc_data)
        with open(dest, "wb") as f:
            f.write(decrypted)
            
        return dest

at_rest_encryption = AtRestEncryption()
