"""Demo mode: answer from the verified example library instead of calling Claude.

For trying the interface without an API key. It only knows the questions in
examples.yaml and eval.yaml (closest match by word overlap); anything else
gets "can't answer". Every response says it came from demo mode.

demo_scenarios.yaml adds scripted *mistakes* (a write request, a wrong-case
filter, a double-counting join) so the safety layers can be demonstrated;
those responses say they are scripted. A second, independent approach is only
ever a scenario's scripted alternative: replaying the same SQL twice would
fake agreement and inflate confidence.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import yaml

from ..generation.llm import SQLResponse
from ..prompt.examples import keywords
from ..prompt.knowledge import translate_sql

MIN_OVERLAP = 0.5


class ReplayGenerator:
    def __init__(self, knowledge_dir: Path, validator=None, dialect=None):
        """`validator` (SQLValidator) fills in the tables/columns used, as a model would;
        `dialect` is the database's, so the library SQL is translated (e.g. for Postgres)."""
        self.validator = validator
        self.dialect = dialect
        knowledge_dir = Path(knowledge_dir)
        pairs: List[Tuple[str, str]] = []
        examples = knowledge_dir / "examples.yaml"
        if examples.exists():
            pairs += [(e["question"], e["sql"]) for e in yaml.safe_load(examples.read_text())]
        evals = knowledge_dir / "eval.yaml"
        if evals.exists():
            pairs += [(c["question"], c["sql"])
                      for c in yaml.safe_load(evals.read_text()).get("cases", []) if c.get("sql")]
        scenarios = knowledge_dir / "demo_scenarios.yaml"
        self.scenarios = {s["question"].strip().lower(): s
                          for s in (yaml.safe_load(scenarios.read_text()) or [])} \
            if scenarios.exists() else {}
        semantic = knowledge_dir / "semantic.yaml"
        self.source_dialect = ((yaml.safe_load(semantic.read_text()) or {}).get("sql_dialect", "sqlite")
                               if semantic.exists() else "sqlite")
        self.library = [(q, s.strip(), keywords(q)) for q, s in pairs]

    def _sql(self, sql: str) -> str:
        return translate_sql(sql, self.source_dialect, self.dialect)

    def generate(self, prompt) -> SQLResponse:
        question = prompt.user.split("## Question\n", 1)[-1].split("\n", 1)[0]
        second_approach = "\n## Approach\n" in prompt.user
        scenario = self.scenarios.get(question.strip().lower())
        if scenario is not None:
            key, note = ("alternative", "alternative_explanation") if second_approach \
                else ("sql", "explanation")
            sql = scenario.get(key)
            if sql:
                sql = self._sql(sql.strip())
                return _response(sql, "Demo scenario (scripted model output): "
                                 + scenario.get(note, ""), 0.8, *self._claims(sql))
        if second_approach:
            return _response(None, "demo mode has no independent second approach for this "
                                   "question", 0.0)
        ambiguous, chosen = _glossary_state(prompt.user)
        if ambiguous:
            # Like the real model: one verified interpretation per definition.
            interpretations = []
            for name in ambiguous:
                match = self._best(f"{question} {name}", must_mention=name)
                if match:
                    interpretations.append({"label": name, "sql": self._sql(match[1]),
                                            "description": f"uses the {name} definition"})
            if len(interpretations) >= 2:
                return SQLResponse.from_json({
                    "sql": None, "explanation": "Demo mode: this term has several meanings.",
                    "tables_used": [], "columns_used": [], "assumptions": [],
                    "confidence": 0.5, "needs_clarification": True,
                    "clarification": {"question": "Which definition do you mean?",
                                      "interpretations": interpretations},
                }, model="demo-replay")
        best, score = None, 0.0
        found = self._best(question, prefer=chosen)
        if found:
            best, score = found[:2], found[2]
        if best is None or score < MIN_OVERLAP:
            return _response(None, "Demo mode only knows the questions in the example "
                                   "library; set ANTHROPIC_API_KEY for real answers.", 0.0)
        sql = self._sql(best[1])
        return _response(sql, f"Demo mode: replayed the verified answer to "
                              f"\"{best[0]}\".", round(score, 2), *self._claims(sql))

    def _claims(self, sql: str):
        """Tables and columns used, as a model would report them."""
        if self.validator is None:
            return [], []
        report = self.validator.validate(sql)
        return report.tables, report.columns


    def _best(self, text: str, must_mention: str = None, prefer: List[str] = ()):
        """(question, sql, score) of the closest library entry, or None."""
        asked = keywords(text)
        best = None
        for q, sql, words in self.library:
            if must_mention and must_mention.lower() not in q.lower():
                continue
            score = len(asked & words) / len(asked | words) if asked | words else 0.0
            if any(p.lower() in q.lower() for p in prefer):
                score += 0.3  # the definition the user already chose
            if best is None or score > best[2]:
                best = (q, sql, score)
        return best


def _glossary_state(prompt_text: str):
    """(definition names of an ambiguous term, names the user chose earlier)."""
    section = prompt_text.split("## Business glossary\n", 1)
    if len(section) < 2:
        return [], []
    body = section[1].split("\n## ", 1)[0]
    ambiguous, chosen = [], []
    for block in re.split(r"\n(?=- )", body):
        names = re.findall(r"^  - ([^:\n]+):", block, re.MULTILINE)
        if "AMBIGUOUS" in block:
            ambiguous = names
        elif "chose this definition earlier" in block:
            chosen += names
    return ambiguous, chosen


def _response(sql, explanation, confidence, tables=(), columns=()) -> SQLResponse:
    return SQLResponse.from_json({
        "sql": sql, "explanation": explanation, "tables_used": list(tables),
        "columns_used": list(columns),
        "assumptions": [], "confidence": confidence, "needs_clarification": False,
        "clarification": None,
    }, model="demo-replay")
