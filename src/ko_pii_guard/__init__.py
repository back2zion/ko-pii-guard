"""Korean PII detection and masking built on Microsoft Presidio."""

from importlib.metadata import version

from ko_pii_guard.analyzer import DEFAULT_ENTITIES, SUPPORTED_ENTITIES, Finding, KoreanPIIGuard

__all__ = ["DEFAULT_ENTITIES", "SUPPORTED_ENTITIES", "Finding", "KoreanPIIGuard"]
__version__ = version("ko-pii-guard")
