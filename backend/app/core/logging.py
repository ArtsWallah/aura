import logging
import sys
from typing import Any

class SensitiveDataFilter(logging.Filter):
    """
    Prevents sensitive tokens, keys, and excessive file dumps from entering standard logs.
    """
    REDACT_KEYS = {"api_key", "password", "token", "secret", "gemini_api_key", "encryption_key"}

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for key in self.REDACT_KEYS:
                if key in record.msg.lower():
                    # Sanitize message
                    record.msg = f"[REDACTED log message containing sensitive reference: {key}]"
        return True

def setup_logger(name: str = "aura") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(SensitiveDataFilter())
        logger.addHandler(console_handler)
        
    return logger

logger = setup_logger()
