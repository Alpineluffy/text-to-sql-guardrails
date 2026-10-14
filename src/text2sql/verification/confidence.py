"""Phase 3 Step 4: one confidence score, from every signal, with reasons.

The score is a weighted average of the signals that were measured. Some
findings override it rather than being averaged in, because a strong signal
elsewhere must never hide them:

  a sanity-check error       (impossible count, value that doesn't exist)  -> at most low
  the approaches disagree    (multi-query)                                  -> at most low
  back-translation mismatch  (SQL answers a different question)            -> at most medium
  something the question requires is missing (coverage)                    -> at most medium
  nothing independently verified (no back-translation, no second query)    -> at most medium

Bands, not decimals, are what users should act on. Weights and thresholds
are starting points: tune them with `python -m text2sql.verification.evaluate`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .backtranslate import BackTranslation
from .coverage import CoverageReport
from .multiquery import MultiQueryCheck
from .sanity import SanityReport


@dataclass(frozen=True)
class ConfidenceConfig:
    w_syntax: float = 0.10
    w_alignment: float = 0.30
    w_sanity: float = 0.20
    w_agreement: float = 0.20
    w_coverage: float = 0.15
    w_model: float = 0.05
    high: float = 0.80
    medium: float = 0.55
    cap_sanity_error: float = 0.45
    cap_disagreement: float = 0.45
    cap_misaligned: float = 0.65
    cap_coverage_missing: float = 0.65
    cap_unverified: float = 0.75


@dataclass
class Confidence:
    score: float
    band: str  # high | medium | low
    problems: List[str]  # what lowers confidence, with specifics
    support: List[str]  # what backs the answer up
    signals: Dict[str, Optional[float]] = field(default_factory=dict)
    caps: List[str] = field(default_factory=list)

    @property
    def reasons(self) -> List[str]:
        return self.problems + self.support


def score_confidence(
    attempts: int = 1,
    backtranslation: Optional[BackTranslation] = None,
    sanity: Optional[SanityReport] = None,
    multiquery: Optional[MultiQueryCheck] = None,
    coverage: Optional[CoverageReport] = None,
    model_confidence: Optional[float] = None,
    config: ConfidenceConfig = ConfidenceConfig(),
) -> Confidence:
    signals: Dict[str, Optional[float]] = {
        "syntax": 1.0 if attempts <= 1 else 0.85,
        "alignment": backtranslation.alignment if backtranslation else None,
        "sanity": sanity.pass_rate if sanity else None,
        "agreement": (None if not multiquery or not multiquery.ran or multiquery.agreed is None
                      else (1.0 if multiquery.agreed else 0.0)),
        "coverage": coverage.score if coverage else None,
        "model": model_confidence,
    }
    weights = {"syntax": config.w_syntax, "alignment": config.w_alignment,
               "sanity": config.w_sanity, "agreement": config.w_agreement,
               "coverage": config.w_coverage, "model": config.w_model}
    measured = {k: v for k, v in signals.items() if v is not None}
    total_weight = sum(weights[k] for k in measured)
    score = sum(weights[k] * v for k, v in measured.items()) / total_weight if total_weight else 0.0

    problems: List[str] = []
    support: List[str] = []
    caps: List[str] = []

    def cap(limit: float, why: str) -> None:
        nonlocal score
        if score > limit:
            score = limit
        caps.append(why)

    if sanity is not None:
        for f in sanity.flags:
            problems.append(f"Result check: {f.message}")
        if sanity.errors:
            cap(config.cap_sanity_error, "sanity-check error")
        elif sanity.checks_run and not sanity.flags:
            n = len(sanity.checks_run)
            support.append("The result check passed" if n == 1 else f"All {n} result checks passed")

    if multiquery is not None and multiquery.ran:
        if multiquery.agreed is False:
            problems.append(f"An independent second query disagrees: {multiquery.explanation}")
            cap(config.cap_disagreement, "approaches disagree")
        elif multiquery.agreed:
            support.append(f"An independent second query agrees ({multiquery.explanation})")

    if backtranslation is not None:
        for m in backtranslation.mismatches:
            problems.append(f"The SQL may not answer the question as asked: {m}")
        if not backtranslation.aligned:
            cap(config.cap_misaligned, "back-translation mismatch")
        else:
            support.append(f"Reading the SQL back gives the same question: "
                           f"\"{backtranslation.back_translated_question}\"")

    if coverage is not None:
        for e in coverage.missing:
            problems.append(f"Not covered: the question needs a query that {e.description}")
        if coverage.missing:
            cap(config.cap_coverage_missing, "missing coverage")
        elif coverage.expectations:
            support.append(f"Uses everything the question mentions "
                           f"({len(coverage.expectations)} check(s))")

    if attempts > 1:
        problems.append(f"The first {attempts - 1} attempt(s) were rejected and repaired")

    independent = signals["alignment"] is not None or signals["agreement"] is not None
    if not independent:
        cap(config.cap_unverified, "not independently verified")
        problems.append("Not independently verified (no back-translation or second query)")

    score = round(max(0.0, min(1.0, score)), 3)
    band = "high" if score >= config.high else "medium" if score >= config.medium else "low"
    return Confidence(score, band, problems, support, signals, caps)
