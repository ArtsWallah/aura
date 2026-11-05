import re
from pathlib import Path
from io import BytesIO
import pandas as pd
from app.core.config import settings
from app.privacy.encryption import encryption_service

def sanitize_filename(filename: str) -> str:
    """
    Remove potentially dangerous characters from uploaded filenames.
    """
    clean_name = re.sub(r"[^\w\.-]", "_", filename)
    clean_name = re.sub(r"\.{2,}", ".", clean_name)
    return clean_name or "unnamed_file"

def validate_safe_path(target_path: str | Path, allow_read_only: bool = True) -> Path:
    """
    Verify that the path is within the allowed workspace / data directories
    to prevent path traversal attacks.

    Enforces pre-resolution symlink checks on all raw path components,
    followed by canonical root containment within UPLOADS_DIR or TRANSFORMS_DIR.

    Platform Limitations:
    On Windows, Python's os.open lacks O_NOFOLLOW without native Win32 API calls
    (CreateFileW with FILE_FLAG_OPEN_REPARSE_POINT). While pre-resolution checks
    and canonical parent verification prevent standard symlink traversal, an
    external TOCTOU race remains theoretically possible if directory components
    are modified concurrently by an outside process.
    """
    target_str = str(target_path)
    if "\x00" in target_str:
        raise PermissionError("Access denied: path contains null byte.")

    raw_path = Path(target_path)

    # Pre-resolution symlink check on raw path and all constituent ancestor components
    curr = raw_path
    while curr != curr.parent:
        if curr.is_symlink():
            raise PermissionError("Access denied: symlinks are not permitted.")
        curr = curr.parent

    path = raw_path.resolve()
    uploads_root = settings.UPLOADS_DIR.resolve()
    transforms_root = settings.TRANSFORMS_DIR.resolve()
    bundled_sample = (settings.DATA_DIR / "sample_sales.csv").resolve()

    is_safe = (path == bundled_sample or uploads_root in path.parents or transforms_root in path.parents)
    if not is_safe or not path.is_file():
        raise PermissionError("Access denied: file is outside managed uploads or transforms.")
    if path != bundled_sample and path.suffix.lower() not in {".enc", ".csv", ".xlsx", ".xls", ".txt", ".md", ".json"}:
        raise PermissionError("Access denied: unsupported managed file type.")
    
    return path


def read_managed_bytes(target_path: str | Path) -> bytes:
    """Read a managed file, decrypting encrypted uploads only in memory."""
    path = validate_safe_path(target_path)
    data = path.read_bytes()
    if path.name.endswith(".enc") or data.startswith(b"gAAAA"):
        data = encryption_service().decrypt_bytes(data)
    elif path.resolve() != (settings.DATA_DIR / "sample_sales.csv").resolve():
        raise PermissionError("Unencrypted legacy uploads require explicit migration before use.")
    return data


def managed_file_suffix(path: str | Path) -> str:
    name = validate_safe_path(path).name
    if name.endswith(".enc"):
        name = name[:-4]
    return Path(name).suffix.lower()


def read_managed_dataset(target_path: str | Path) -> pd.DataFrame:
    suffix = managed_file_suffix(target_path)
    source = BytesIO(read_managed_bytes(target_path))
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(source)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    raise ValueError(f"Unsupported tabular format: '{suffix}'.")

def mask_secret(secret: str | None, visible_suffix_len: int = 4) -> str:
    """
    Mask an API key or token for display in UI or logs.
    """
    return "Configured" if secret else "None"

