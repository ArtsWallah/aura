import re
from typing import Tuple

class DataRedactor:
    """Best-effort recognizer for known patterns, not a general PII detector.

    Hybrid egress must not rely on this class to decide whether arbitrary text
    is safe. The outbound client fails closed for Hybrid until an explicit
    release/approval contract is available.
    """
    PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "IP_ADDRESS": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
        "API_KEY": r"\b(?:sk-[a-zA-Z0-9]{20,}|AIza[0-9A-Za-z-_]{35})\b"
    }

    def redact_text(self, text: str) -> Tuple[str, int]:
        redacted_text = text
        total_redacted = 0
        
        for entity_type, pattern in self.PATTERNS.items():
            matches = list(re.finditer(pattern, redacted_text))
            if matches:
                total_redacted += len(matches)
                redacted_text = re.sub(pattern, f"[REDACTED_{entity_type}]", redacted_text)
                
        return redacted_text, total_redacted

data_redactor = DataRedactor()

