"""Measure accuracy and whether the confidence score can be trusted.

    python -m text2sql.verification.evaluate            # asks before spending API credits
    python -m text2sql.verification.evaluate --limit 5 --yes

Runs every question in knowledge/<db>/eval.yaml through the full pipeline
and compares the result (not the SQL) with the verified answer. Reports:

  accuracy      share of questions answered correctly
  calibration   accuracy within each confidence band; "high" should be
                clearly more accurate than "medium", and "medium" than "low"
  detection     share of wrong answers that were NOT marked high
  false alarms  share of right answers marked low

Use the per-question report to tune ConfidenceConfig.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .multiquery import compare_results

BANDS = ("high", "medium", "low", "none")


@dataclass
class EvalCase:
    question: str
    sql: str
    choice: Optional[str] = None  # which option to pick if asked to clarify
    tags: List[str] = field(default_factory=list)


@dataclass
class CaseResult:
    question: str
    tags: List[str]
    status: str
    correct: bool
    band: str
    confidence: Optional[float]
    explanation: str
    sql: Optional[str]
    problems: List[str]


@dataclass
class EvalReport:
    results: List[CaseResult]

    @property
    def accuracy(self) -> float:
        return _share(self.results, lambda r: r.correct)

    def by_band(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for band in BANDS:
            rows = [r for r in self.results if r.band == band]
            if rows:
                out[band] = {"n": len(rows), "accuracy": _share(rows, lambda r: r.correct)}
        return out

    @property
    def detection(self) -> Optional[float]:
        wrong = [r for r in self.results if not r.correct]
        return _share(wrong, lambda r: r.band != "high") if wrong else None

    @property
    def false_alarms(self) -> Optional[float]:
        right = [r for r in self.results if r.correct]
        return _share(right, lambda r: r.band == "low") if right else None

    def summary(self) -> str:
        lines = [f"Accuracy: {self.accuracy:.0%} ({sum(r.correct for r in self.results)}"
                 f"/{len(self.results)})", "", "Calibration (accuracy per confidence band):"]
        for band, stats in self.by_band().items():
            lines.append(f"  {band:<7} {stats['n']:>3} question(s)   {stats['accuracy']:.0%} correct")
        if self.detection is not None:
            lines.append(f"\nWrong answers not marked high (detection): {self.detection:.0%}")
        if self.false_alarms is not None:
            lines.append(f"Right answers marked low (false alarms):   {self.false_alarms:.0%}")
        misses = [r for r in self.results if not r.correct]
        if misses:
            lines.append("\nWrong answers:")
            for r in misses:
                lines.append(f"  [{r.band}] {r.question}  ({r.status}: {r.explanation})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"accuracy": self.accuracy, "by_band": self.by_band(),
                "detection": self.detection, "false_alarms": self.false_alarms,
                "results": [asdict(r) for r in self.results]}


def load_eval(path: Path):
    """The answerable cases of the golden set (see text2sql.evaluation for all of it)."""
    data = yaml.safe_load(Path(path).read_text())
    today = data.get("today")
    if isinstance(today, str):
        today = date.fromisoformat(today)
    cases = [EvalCase(c["question"], c["sql"].strip(), c.get("choice"), c.get("tags", []))
             for c in data["cases"] if c.get("expect", "answer") == "answer"]
    return today, cases


def run_eval(service: Any, cases: List[EvalCase], gold_executor: Any,
             progress: bool = False) -> EvalReport:
    from ..execution import ClarificationResponse

    results = []
    for i, case in enumerate(cases, 1):
        gold = gold_executor.run(case.sql)
        response = service.ask(case.question)
        if isinstance(response, ClarificationResponse):
            labels = [o.label.lower() for o in response.options]
            pick = labels.index(case.choice.lower()) if case.choice and \
                case.choice.lower() in labels else 0
            response = service.choose(response.request_id, pick)
        if response.ok:
            comparison = compare_results(gold.columns, gold.rows, gold.truncated,
                                         response.columns, response.rows, response.truncated)
            correct, explanation = bool(comparison.agree), comparison.explanation
        else:
            correct, explanation = False, response.error or response.status
        report = response.verification
        results.append(CaseResult(
            question=case.question, tags=case.tags, status=response.status, correct=correct,
            band=response.confidence_band or "none", confidence=response.confidence,
            explanation=explanation, sql=response.sql,
            problems=list(report.confidence.problems) if report else [],
        ))
        if progress:
            mark = "ok  " if correct else "MISS"
            print(f"{mark} [{results[-1].band:<6}] {i}/{len(cases)} {case.question}", flush=True)
    return EvalReport(results)


def _share(rows, predicate) -> float:
    return sum(1 for r in rows if predicate(r)) / len(rows) if rows else 0.0


def main() -> None:
    import sqlalchemy as sa

    from ..execution import create_service
    from ..sandbox import SandboxedExecutor

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-url", default="sqlite:///data/ecommerce.db")
    parser.add_argument("--knowledge", type=Path, default=Path("knowledge/ecommerce"))
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--no-verify", action="store_true", help="accuracy only, no confidence")
    parser.add_argument("--multi-query", default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--yes", action="store_true", help="don't ask before calling the API")
    args = parser.parse_args()

    today, cases = load_eval(args.knowledge / "eval.yaml")
    cases = cases[: args.limit] if args.limit else cases
    calls = len(cases) * (1 if args.no_verify else 3.5)
    print(f"{len(cases)} questions, roughly {calls:.0f} model calls (claude-opus-5-5).")
    if not args.yes and input("Run it? [y/N] ").strip().lower() != "y":
        sys.exit("Cancelled.")

    engine = sa.create_engine(args.db_url)
    service = create_service(engine, knowledge_dir=args.knowledge, today=today,
                             verify=not args.no_verify, multi_query=args.multi_query)
    report = run_eval(service, cases, SandboxedExecutor(engine), progress=True)
    print("\n" + report.summary())

    out = Path("cache/eval") / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()
