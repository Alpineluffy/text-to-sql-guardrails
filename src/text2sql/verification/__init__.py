"""Phase 3: detect answers that look right but aren't, and score confidence."""
from .backtranslate import BackTranslation, BackTranslator
from .confidence import Confidence, ConfidenceConfig, score_confidence
from .coverage import CoverageChecker, CoverageReport, Expectation
from .multiquery import MultiQueryCheck, MultiQueryChecker, compare_results, complexity
from .sanity import ResultSanityChecker, SanityFlag, SanityReport
from .verifier import VerificationReport, Verifier

__all__ = [
    "BackTranslation",
    "BackTranslator",
    "Confidence",
    "ConfidenceConfig",
    "CoverageChecker",
    "CoverageReport",
    "Expectation",
    "MultiQueryCheck",
    "MultiQueryChecker",
    "ResultSanityChecker",
    "SanityFlag",
    "SanityReport",
    "VerificationReport",
    "Verifier",
    "compare_results",
    "complexity",
    "score_confidence",
]
