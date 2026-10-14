# Evaluation

Reproduce: `python -m text2sql.evaluation offline` (no API key) and `python -m text2sql.evaluation live` (calls Claude; asks first).

## Golden dataset

56 questions in `knowledge/ecommerce/eval.yaml`: 45 answerable, 5 ambiguous, 6 unanswerable. Every verified query runs and returns rows; on sqlite, postgresql the results are identical.

## Hallucination detection: mutation suite

Each verified query is broken on purpose and the deterministic checks (result sanity + coverage; no API calls) must flag it. A mutant only counts if its result really differs from the correct one.

| Bug injected | Detected |
|---|---|
| drop_date | 11/11 (100%) |
| drop_limit | 7/7 (100%) |
| drop_status | 12/14 (86%) |
| fanout_join | 9/9 (100%) |
| value_case | 23/23 (100%) |
| wrong_agg | 0/15 (0%) |
| wrong_value | 23/23 (100%) |

Overall: **83%** of 102; false alarms on the 45 correct queries: **0**.

`wrong_agg` (e.g. SUM where AVG was meant) can't be caught by checking data: both results are plausible. That class is covered by back-translation, which needs the API and is measured in the live run.

## Guardrail effectiveness

- SQL attacks refused: **38/38**; Postgres-only attacks: **21/21**
- Unsafe queries executed: **0**; database fingerprint unchanged: **yes**
- With the guardrails switched off, the database layer alone refuses 36/37 (allowed: 'BEGIN TRANSACTION', which can't change data on a read-only connection)
- Legitimate queries wrongly blocked: **0/61**

## Live run

Not measured yet (needs an API key: `python -m text2sql.evaluation live`).
