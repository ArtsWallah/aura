import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.core.config import settings
from app.core.security import sanitize_filename
from app.privacy.encryption import at_rest_encryption

router = APIRouter(prefix="/files", tags=["Files"])

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    encrypt_at_rest: bool = Form(default=False)
):
    clean_name = sanitize_filename(file.filename or "upload.csv")
    ext = Path(clean_name).suffix.lower()
    
    allowed_exts = {".csv", ".xlsx", ".xls", ".txt", ".md", ".json"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(allowed_exts)}"
        )

    out_path = settings.UPLOADS_DIR / clean_name
    
    # Save file
    with open(out_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    size_bytes = out_path.stat().st_size
    encrypted_path = None
    
    if encrypt_at_rest:
        enc_file = at_rest_encryption.encrypt_file(out_path)
        encrypted_path = str(enc_file)

    return {
        "file_name": clean_name,
        "file_path": str(out_path),
        "relative_path": f"data/uploads/{clean_name}",
        "size_bytes": size_bytes,
        "format": ext[1:].upper(),
        "encrypted_at_rest": encrypt_at_rest,
        "encrypted_file_path": encrypted_path
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
                "modified_at": f.stat().st_mtime
            })
    return files

