import os
from pathlib import Path
from typing import Literal
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
    ENCRYPTION_KEY: str = "aura-dev-secret-key-32-chars-ok!" # In prod, provide 32-byte urlsafe base64
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
