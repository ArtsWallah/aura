import re
from pathlib import Path
from app.core.config import settings

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
    """
    path = Path(target_path).resolve()
    allowed_roots = [
        settings.DATA_DIR.resolve(),
        settings.BASE_DIR.resolve(),
    ]
    
    is_safe = any(path == root or root in path.parents for root in allowed_roots)
    if not is_safe:
        raise PermissionError(f"Access denied: Path '{target_path}' is outside permitted directory boundaries.")
    
    return path

def mask_secret(secret: str | None, visible_suffix_len: int = 4) -> str:
    """
    Mask an API key or token for display in UI or logs.
    """
    if not secret:
        return "None"
    if len(secret) <= visible_suffix_len:
        return "••••"
    return "••••" + secret[-visible_suffix_len:]

