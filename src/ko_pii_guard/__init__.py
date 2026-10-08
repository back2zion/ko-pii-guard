"""Korean PII detection and masking built on Microsoft Presidio."""

from ko_pii_guard.analyzer import DEFAULT_ENTITIES, Finding, KoreanPIIGuard

__all__ = ["DEFAULT_ENTITIES", "Finding", "KoreanPIIGuard"]
__version__ = "0.1.0"
