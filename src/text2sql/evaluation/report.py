"""Turn saved evaluation results into docs/EVAL.md and the README's results block.

Only measured numbers are published: anything not yet run says so.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

RESULTS_DIR = Path("docs/eval")
START, END = "<!-- EVAL:START -->", "<!-- EVAL:END -->"
NOT_MEASURED = "not measured yet (needs an API key: `python -m text2sql.evaluation live`)"


def load(name: str) -> Optional[dict]:
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def composition(by_expect: dict) -> str:
    names = {"answer": "answerable", "clarify": "ambiguous", "cannot_answer": "unanswerable"}
    return ", ".join(f"{n} {names.get(k, k)}" for k, n in by_expect.items())


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def readme_block(offline: Optional[dict], live: Optional[dict]) -> str:
    lines = [START, "## Results", ""]
    if offline:
        m, s = offline["mutation"], offline["security"]
        attacks = s["attacks"] + s["pg_attacks"]
        refused = s["refused"] + s["pg_refused"]
        executed = len(s["executed"])
        if live:
            lines.append(f"**{pct(live['execution_accuracy'])} execution accuracy, "
                         f"{pct(m['detection_rate'])} of injected bugs detected, "
                         f"{executed} unsafe queries executed across {attacks} attacks.**")
        else:
            lines.append(f"**{pct(m['detection_rate'])} of injected bugs detected with "
                         f"{pct(m['false_alarm_rate'])} false alarms; {executed} unsafe queries "
                         f"executed across {attacks} attacks.** Execution accuracy: {NOT_MEASURED}.")
        lines += ["", "| Metric | Result |", "|---|---|"]
        if live:
            b = live["behaviour"]
            lines += [
                f"| Execution accuracy ({b['answer']['n']} answerable questions) | "
                f"{pct(live['execution_accuracy'])} |",
                f"| SQL exact match after normalising (secondary) | {pct(live['exact_match'])} |",
            ]
            if "clarify" in b:
                lines.append(f"| Asks on ambiguous questions | {b['clarify']['ok']}/{b['clarify']['n']} |")
            if "cannot_answer" in b:
                lines.append(f"| Declines unanswerable questions | "
                             f"{b['cannot_answer']['ok']}/{b['cannot_answer']['n']} |")
            if live.get("detection") is not None:
                lines.append(f"| Wrong live answers kept out of \"high\" confidence | "
                             f"{pct(live['detection'])} of {live['wrong_answers']} |")
            lines.append(f"| Prompt-injection questions handled safely | "
                         f"{live['injections_safe']}/{len(live['injections'])} |")
        else:
            lines.append(f"| Execution accuracy | {NOT_MEASURED} |")
        lines += [
            f"| Injected bugs detected (mutation suite, no API) | {pct(m['detection_rate'])} "
            f"of {m['mutants']} |",
            f"| False alarms on correct queries | {pct(m['false_alarm_rate'])} of {m['gold_checked']} |",
            f"| Attacks refused | {refused}/{attacks} |",
            f"| Unsafe queries executed | {executed} (database unchanged: "
            f"{'yes' if s['database_unchanged'] else 'NO'}) |",
            f"| Legitimate queries wrongly blocked | {len(s['false_blocks'])}/{s['legitimate']} |",
        ]
        lines += ["", f"Golden set: {offline['dataset']['total']} questions "
                      f"({composition(offline['dataset']['by_expect'])}). Measured {offline['measured_at']}"
                      + (f" (offline) and {live['measured_at']} (live, {live['model']})" if live else "")
                      + ". Details: [docs/EVAL.md](docs/EVAL.md)."]
    else:
        lines.append("Not measured yet: run `python -m text2sql.evaluation offline`.")
    lines.append(END)
    return "\n".join(lines)


def eval_markdown(offline: Optional[dict], live: Optional[dict]) -> str:
    out = ["# Evaluation", "",
           "Reproduce: `python -m text2sql.evaluation offline` (no API key) and "
           "`python -m text2sql.evaluation live` (calls Claude; asks first).", ""]
    if offline:
        d = offline["dataset"]
        out += ["## Golden dataset", "",
                f"{d['total']} questions in `knowledge/ecommerce/eval.yaml`: {composition(d['by_expect'])}. "
                f"Every verified query runs and returns rows"
                + (f"; on {', '.join(d['databases'])} the results are identical"
                   if len(d["databases"]) > 1 else "") + ".", ""]
        m = offline["mutation"]
        out += ["## Hallucination detection: mutation suite", "",
                "Each verified query is broken on purpose and the deterministic checks "
                "(result sanity + coverage; no API calls) must flag it. A mutant only counts "
                "if its result really differs from the correct one.", "",
                "| Bug injected | Detected |", "|---|---|"]
        for kind, v in m["by_kind"].items():
            out.append(f"| {kind} | {v['detected']}/{v['n']} ({pct(v['rate'])}) |")
        out += ["", f"Overall: **{pct(m['detection_rate'])}** of {m['mutants']}; "
                    f"false alarms on the {m['gold_checked']} correct queries: "
                    f"**{len(m['false_alarms'])}**.", "",
                "`wrong_agg` (e.g. SUM where AVG was meant) can't be caught by checking data: "
                "both results are plausible. That class is covered by back-translation, which "
                "needs the API and is measured in the live run.", ""]
        s = offline["security"]
        out += ["## Guardrail effectiveness", "",
                f"- SQL attacks refused: **{s['refused']}/{s['attacks']}**; Postgres-only "
                f"attacks: **{s['pg_refused']}/{s['pg_attacks']}**",
                f"- Unsafe queries executed: **{len(s['executed'])}**; database fingerprint "
                f"unchanged: **{'yes' if s['database_unchanged'] else 'NO'}**",
                f"- With the guardrails switched off, the database layer alone refuses "
                f"{s['db_layer_refused']}/{s['db_layer_checked']}"
                + (f" (allowed: {', '.join(repr(x) for x in s['db_layer_allowed'])}, which "
                   f"can't change data on a read-only connection)" if s["db_layer_allowed"] else ""),
                f"- Legitimate queries wrongly blocked: **{len(s['false_blocks'])}/{s['legitimate']}**",
                ""]
    if live:
        out += ["## Live run", "", f"Model: {live['model']}, measured {live['measured_at']}.", "",
                f"- Execution accuracy: **{pct(live['execution_accuracy'])}**",
                f"- SQL exact match (normalised): {pct(live['exact_match'])}"]
        for kind, v in live["behaviour"].items():
            out.append(f"- Correct behaviour, {kind}: {v['ok']}/{v['n']}")
        out += ["", "| Confidence band | Questions | Accuracy |", "|---|---|---|"]
        for band, v in live["calibration"].items():
            out.append(f"| {band} | {v['n']} | {pct(v['accuracy'])} |")
        out += ["", f"Prompt injection: {live['injections_safe']}/{len(live['injections'])} handled "
                    f"safely; database unchanged: {live['database_unchanged']}.", ""]
        misses = [r for r in live["results"]
                  if not r["behaviour_ok"] or r["correct"] is False]
        if misses:
            out += ["### Misses", ""]
            for r in misses:
                out.append(f"- [{r['band']}] ({r['expect']}) {r['question']}: {r['detail']}")
    else:
        out += ["## Live run", "", NOT_MEASURED[0].upper() + NOT_MEASURED[1:] + "."]
    return "\n".join(out) + "\n"


def write_reports(readme: Path = Path("README.md"), doc: Path = Path("docs/EVAL.md")) -> None:
    offline, live = load("offline"), load("live")
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(eval_markdown(offline, live))
    block = readme_block(offline, live)
    text = readme.read_text() if readme.exists() else ""
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    else:  # first time: right under the title, so the numbers lead
        lines = text.splitlines()
        insert = 1 if lines and lines[0].startswith("# ") else 0
        lines[insert:insert] = ["", block]
        text = "\n".join(lines) + "\n"
    readme.write_text(text)
