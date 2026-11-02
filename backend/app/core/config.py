from pathlib import Path
import ipaddress
from typing import Literal
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR.parent / "data"

class Settings(BaseSettings):
    PROJECT_NAME: str = "AURA - Autonomous Understanding, Reasoning & Analytics"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    # Environment
    ENVIRONMENT: Literal["development", "production", "test"] = "development"
    DEBUG: bool = True
    
    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # LLM Settings
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.0-flash"
    DEFAULT_LLM_PROVIDER: str = "gemini"
    LLM_TIMEOUT_SECONDS: int = 45
    LLM_TEMPERATURE: float = 0.2
    
    # Privacy Modes: local_only, hybrid, cloud_enabled
    DEFAULT_PRIVACY_MODE: Literal["local_only", "hybrid", "cloud_enabled"] = "hybrid"
    # Must be a URL-safe base64 Fernet key (generate with Fernet.generate_key()).
    ENCRYPTION_KEY: str | None = None
    MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024
    DATA_RETENTION_HOURS: int = 72
    
    # Directories
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    CHARTS_DIR: Path = DATA_DIR / "charts"
    REPORTS_DIR: Path = DATA_DIR / "reports"
    EXPORTS_DIR: Path = DATA_DIR / "exports"
    
    # Database
    DATABASE_URL: str = f"sqlite:///{DATA_DIR / 'aura.db'}"

    @field_validator("HOST")
    @classmethod
    def require_loopback_host(cls, value: str) -> str:
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError("HOST must be a loopback IP address.") from exc
        if not address.is_loopback:
            raise ValueError("HOST must be a loopback address for the supported single-user deployment.")
        return value
    
    # Agent Execution
    MAX_PLAN_STEPS: int = 15
    MAX_STEP_RETRIES: int = 2
    STEP_TIMEOUT_SECONDS: int = 60
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def init_directories(self):
        for directory in [self.DATA_DIR, self.UPLOADS_DIR, self.CHARTS_DIR, self.REPORTS_DIR, self.EXPORTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.init_directories()

