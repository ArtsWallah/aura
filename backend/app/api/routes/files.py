import uuid
import json
import os
from io import BytesIO
from pathlib import Path
from zipfile import is_zipfile

import pandas as pd

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import settings
from app.core.security import sanitize_filename
from app.privacy.encryption import encryption_service

router = APIRouter(prefix="/files", tags=["Files"])
ALLOWED_EXTS = {".csv", ".xlsx", ".xls", ".txt", ".md", ".json"}


def _validate_content(ext: str, raw: bytes) -> None:
    try:
        if ext == ".csv":
            pd.read_csv(BytesIO(raw), nrows=5)
        elif ext == ".xlsx" and not is_zipfile(BytesIO(raw)):
            raise ValueError("Invalid XLSX container.")
        elif ext == ".xls" and not raw.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
            raise ValueError("Invalid XLS workbook signature.")
        elif ext in {".txt", ".md"}:
            raw.decode("utf-8")
        elif ext == ".json":
            json.loads(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="File content does not match the selected file type.") from exc


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    clean_name = sanitize_filename(file.filename or "upload.csv")
    ext = Path(clean_name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    try:
        cipher = encryption_service()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Encrypted uploads are unavailable: configure a valid ENCRYPTION_KEY.") from exc

    raw = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(raw) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_BYTES} byte limit.")
    _validate_content(ext, raw)

    # Opaque unique names avoid collisions and do not expose user filenames on disk.
    stored_name = f"upload_{uuid.uuid4().hex}{ext}.enc"
    out_path = settings.UPLOADS_DIR / stored_name
    tmp_path = settings.UPLOADS_DIR / f".{stored_name}.tmp"
    created_tmp_identity: tuple[int, int] | None = None
    try:
        # Exclusive creation prevents a collision from replacing an existing upload.
        with tmp_path.open("xb") as stream:
            created_stat = os.fstat(stream.fileno())
            created_tmp_identity = (created_stat.st_dev, created_stat.st_ino)
            stream.write(cipher.encrypt_bytes(raw))
            stream.flush()
        if out_path.exists():
            raise HTTPException(status_code=409, detail="Upload identifier collision; retry the upload.")
        tmp_path.replace(out_path)
    finally:
        if created_tmp_identity is not None:
            try:
                current_stat = tmp_path.lstat()
                if (current_stat.st_dev, current_stat.st_ino) == created_tmp_identity:
                    tmp_path.unlink()
            except FileNotFoundError:
                pass

    return {
        "file_name": clean_name,
        "file_path": str(out_path),
        "relative_path": f"data/uploads/{stored_name}",
        "size_bytes": len(raw),
        "format": ext[1:].upper(),
        "encrypted_at_rest": True,
    }


@router.get("/list")
def list_uploaded_files():
    files = []
    for f in settings.UPLOADS_DIR.iterdir():
        if f.is_file() and not f.name.startswith("."):
            files.append({
                "file_name": f.name,
                "file_path": str(f),
                "relative_path": f"data/uploads/{f.name}",
                "size_bytes": f.stat().st_size,
                "modified_at": f.stat().st_mtime,
                "encrypted_at_rest": f.name.endswith(".enc") or _is_encrypted(f),
                "requires_migration": not (f.name.endswith(".enc") or _is_encrypted(f)),
            })
    return files


def _is_encrypted(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(5) == b"gAAAA"
