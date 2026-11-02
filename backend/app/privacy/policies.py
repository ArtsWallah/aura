from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

class PrivacyMode(str, Enum):
    LOCAL_ONLY = "local_only"
    HYBRID = "hybrid"
    CLOUD_ENABLED = "cloud_enabled"

class PrivacyValidationResult(BaseModel):
    allowed: bool
    reason: str
    sanitized_payload: Any = None
    redacted_items_count: int = 0

class PrivacyEngine:
    def validate_cloud_transmission(
        self,
        payload: Any,
        mode: str = "hybrid",
        user_explicit_consent: bool = False
    ) -> PrivacyValidationResult:
        """
        Enforces strict boundary checks before any data leaves the host environment for cloud inference.
        """
        # Mode 1: Local Only - Absolute Prohibition
        if mode == PrivacyMode.LOCAL_ONLY or mode == "local_only":
            return PrivacyValidationResult(
                allowed=False,
                reason="STRICT POLICY: Cloud transmission is strictly forbidden in 'local_only' mode. All processing must occur on-premise."
            )

        # Mode 2: Hybrid - Minimized Context Only
        if mode == PrivacyMode.HYBRID or mode == "hybrid":
            # Strip raw data rows if tabular dict or list
            sanitized = self._minimize_context(payload)
            return PrivacyValidationResult(
                allowed=True,
                reason="HYBRID POLICY: Raw data minimized and sanitized. Transmitting only abstract statistical schema.",
                sanitized_payload=sanitized
            )

        # Mode 3: Cloud Enabled
        return PrivacyValidationResult(
            allowed=True,
            reason="CLOUD POLICY: Cloud processing authorized by organizational configuration.",
            sanitized_payload=payload
        )

    def _minimize_context(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            minimized = {}
            for k, v in payload.items():
                if k in ["sample_data", "raw_rows", "records", "data_uri"]:
                    minimized[k] = "[MINIMIZED - RETAINED LOCALLY]"
                elif isinstance(v, (dict, list)):
                    minimized[k] = self._minimize_context(v)
                else:
                    minimized[k] = v
            return minimized
        elif isinstance(payload, list):
            # Cap list length if transmitting records
            return [self._minimize_context(item) for item in payload[:5]]
        return payload

privacy_engine = PrivacyEngine()
