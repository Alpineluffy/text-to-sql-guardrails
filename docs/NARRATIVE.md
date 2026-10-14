# Project narrative

How to describe this project: safety first, every number measured and
scoped to how it was measured.

## The headline

> I built a text-to-SQL system that blocked **100% of destructive operations**
> in a 59-attack test suite, with zero unsafe queries reaching the database,
> and that flags **83% of injected SQL errors** before they reach the user,
> with zero false alarms on correct queries.

Once the live evaluation has run (`python -m text2sql.evaluation live`), lead
with all three:

> I built a text-to-SQL system with **X% execution accuracy** that blocks
> **100% of destructive operations** and flags **83% of hallucinated queries**
> before they reach the user.

Until then, don't quote an accuracy figure: the README says "not measured yet",
and the narrative should match it.

## 30-second version

Language models write SQL that looks right and is quietly wrong: a filter on
`'delivered'` when the data says `'DELIVERED'` returns zero, and a careless join
triples revenue. Worse, they'll write a `DELETE` if asked nicely. I built a
text-to-SQL system that treats the model as untrusted. Its SQL is checked like
user input and run in a sandbox that physically can't write, and every answer
is cross-examined (result sanity checks, an independent second query, and a
model that describes the SQL without seeing the question) before the user sees
a confidence band with specific reasons. I measured all of it: 59 attacks, none
executed; 83% of injected errors caught with no false alarms.

## Résumé bullets

- Built a text-to-SQL system (Python, Claude, FastAPI, Streamlit, PostgreSQL,
  Docker) with defence in depth: an allow-list guardrail layer plus a
  read-only database role with column-level permissions; **0 of 59 attacks
  executed**, verified by fingerprinting the database before and after.
- Designed hallucination detection (result sanity checks, join fan-out
  measured directly, back-translation, independent second query) that flags
  **83% of injected SQL errors with zero false alarms** on 45 verified queries,
  measured by mutation testing.
- Built a 56-question golden evaluation set (incl. ambiguous and unanswerable
  questions) verified on SQLite and PostgreSQL, and an evaluation suite that
  publishes its own results to the README.
- Shipped an API and UI with clarifying questions, editable SQL that goes
  through the same safety layers, and a feedback loop with a review gate so
  user feedback can't poison the prompt.

## Stories for interviews

**The DELETE that looked like a SELECT.** A PostgreSQL query like
`WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d` is a SELECT at the
top level, so a guardrail that checks the statement type passes it. That's why
the guardrail inspects every node of the parsed query, and why there's a
second line of defence: the database role can't write at all. I test that
second line with the guardrails switched off.

**Finding blind spots with mutation testing.** A model that is usually right
makes too few mistakes to measure detection on, so I injected realistic bugs
into correct queries. The first run found that "Which country has *the most*
customers?" without `LIMIT 1` slipped through. Fixing that created a false
alarm on "Which *3* customers…", which the same suite caught. The suite also
showed that a "sum exceeds the table total" check misses filtered queries, so
I replaced it with a direct measurement: re-run the query's joins and compare
joined rows with distinct keys.

**SQLite and PostgreSQL disagreeing on 2.565.** The cross-database check
found the golden answer 2.565 rounding to 2.56 on SQLite (binary float) and
2.57 on PostgreSQL (exact decimal). Both are right, so result comparison now
allows rounding-level differences; otherwise correct answers would have been
scored wrong.

**Honest confidence.** An average can hide a fatal problem, so serious
findings cap the band (an impossible result is never better than "low"), and
nothing is "high" unless something independent confirmed it. The demo runs
without an API key, so its answers are honestly "medium".

## Numbers and where they come from

| Claim | Source | Scope |
|---|---|---|
| 100% of destructive operations blocked | `docs/eval/offline.json` → security | 38 SQLite + 21 PostgreSQL attacks; database fingerprinted before/after |
| 0 legitimate queries wrongly blocked | same | 12 hand-written + 49 golden queries |
| 83% of injected errors flagged, 0 false alarms | `docs/eval/offline.json` → mutation | 102 injected bugs across 45 verified queries; deterministic checks only (no API) |
| Execution accuracy | `docs/eval/live.json` (not yet produced) | Needs `python -m text2sql.evaluation live` with an API key |

What not to claim: "100% safe" in general (the claim is about this attack
suite), or that 83% is the rate on *live* model mistakes (it's on injected
ones; the live run measures real mistakes).
