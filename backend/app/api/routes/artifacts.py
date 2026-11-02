from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.core.config import settings
from app.privacy.encryption import encryption_service
from app.privacy.encryption import migrate_plaintext_files

router = APIRouter(prefix="/artifacts", tags=["Artifacts"])


@router.get("/{filename}")
def get_artifact(filename: str):
    # Path.name check rejects traversal and separators on Windows and POSIX.
    if Path(filename).name != filename or not filename.endswith(".enc"):
        raise HTTPException(status_code=404, detail="Artifact not found.")
    encrypted = filename.endswith(".enc")
    if filename.startswith("chart_") and (filename.endswith(".png.enc") or filename.endswith(".png")):
        directory, media_type = settings.CHARTS_DIR, "image/png"
    elif filename.startswith("report_") and filename.endswith((".md.enc", ".html.enc", ".md", ".html")):
        directory = settings.REPORTS_DIR
        media_type = "text/markdown" if ".md" in filename else "text/html"
    else:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    path = directory / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    try:
        encrypted_data = path.read_bytes()
        if not encrypted and not encrypted_data.startswith(b"gAAAA"):
            raise HTTPException(status_code=404, detail="Legacy artifact requires migration.")
        content = encryption_service().decrypt_bytes(encrypted_data)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Artifact decryption unavailable.") from exc
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
    if media_type == "text/html":
        headers["Content-Security-Policy"] = "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; sandbox"
    return Response(content=content, media_type=media_type, headers=headers)


@router.post("/migrate-legacy")
def migrate_legacy_artifacts():
    """Explicitly encrypt legacy uploads, reports, and charts in place."""
    try:
        migrated = []
        for directory in (settings.UPLOADS_DIR, settings.CHARTS_DIR, settings.REPORTS_DIR):
            migrated.extend(f"{directory.name}/{name}" for name in migrate_plaintext_files(list(directory.iterdir())))
    except (RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Legacy artifact migration failed; original files were preserved where not yet replaced.") from exc
    return {"migrated_files": migrated, "count": len(migrated)}
