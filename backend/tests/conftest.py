import os

from cryptography.fernet import Fernet

# Per-process test key only. Production must configure its own persistent key.
os.environ.setdefault("ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
# This host session exports DEBUG=release; force a valid test setting before
# application settings are imported.
os.environ["DEBUG"] = "false"
