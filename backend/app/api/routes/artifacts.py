from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.privacy.encryption import migrate_plaintext_files

router = APIRouter(prefix="/artifacts", tags=["Artifacts"])


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
