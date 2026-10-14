# Development log

How the system was built, phase by phase, with the design notes from each step. Commands are run from the repository root. For the overview, see the [README](../README.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Step 1: Schema extraction (done)

Introspects a database with SQLAlchemy and saves a JSON description that later
steps use as LLM context.

```bash
.venv/bin/python -m text2sql.sample_data          # builds data/ecommerce.db
.venv/bin/python -m text2sql.schema               # extracts -> cache/schema/ecommerce.json
.venv/bin/python -m text2sql.schema --db-url postgresql://user@host/db --namespace public
```

What it captures, and the hallucination each item prevents:

| Captured | Prevents |
|---|---|
| Tables, views, columns, types, PKs, comments | Made-up tables/columns |
| Declared foreign keys + **inferred** ones (name match, verified against data) | Wrong join keys |
| Categorical values with frequency order and case style (`DELIVERED`, `web`) | Wrong filter values |
| Value index for high-cardinality text (`Laptop Pro 14`) | Missed entities in schema filtering (Step 3) |
| Min/max for numbers and dates | Impossible ranges, wrong date filters |
| PII columns flagged; their values are never read | Leaking personal data into prompts |

Results are cached and reused until the structure (or config) changes.
Data-only changes aren't detected, so use `--refresh` or `--max-age-hours`.

Code: `src/text2sql/schema/`. Tests: `.venv/bin/python -m pytest`.

## Step 2: Prompt constructor (done)

Builds the prompt for one question from the extracted schema plus hand-written
knowledge in `knowledge/<db>/`:

- `semantic.yaml`: table/column descriptions, synonyms and the business glossary
  (a term may have several definitions, e.g. gross vs net revenue)
- `examples.yaml`: verified question→SQL pairs (each one is executed by the tests)

```bash
.venv/bin/python -m text2sql.prompt "What was revenue from india last quarter?"
.venv/bin/python -m text2sql.prompt "Net revenue by segment" --tables orders,customers --system
```

| Prompt section | Source |
|---|---|
| System: rules, dialect notes, response format | Fixed per database, so it can be prompt-cached |
| Schema as annotated `CREATE TABLE` | Step 1 profile + descriptions; PII marked, values hidden |
| Relationships | Declared + inferred FKs among the included tables |
| Values mentioned in the question | `india` → `customers.country = 'India'` |
| Business glossary | Only the matched terms; ambiguous ones are flagged |
| Date context | "last quarter" etc. resolved from today's date |
| Examples | 3 closest by word overlap, skipping ones that would bias an ambiguous term |

The knowledge files are validated against the schema, so a renamed column
fails loudly. If the prompt goes over the token budget, examples are dropped
first, then value lists are shortened.

## Step 3: Schema filtering (done)

Picks the tables a question needs, so large schemas don't flood the prompt.
Uses a local embedding model (`all-MiniLM-L6-v2`, ~90 MB, downloaded once).

```bash
.venv/bin/pip install -e ".[embeddings]"
.venv/bin/python -m text2sql.prompt --filter "Revenue by category for Japanese clients"
.venv/bin/python -m text2sql.retrieval.evaluate           # recall on the labelled set
.venv/bin/python -m text2sql.retrieval.evaluate --sweep   # compare thresholds
```

A table is selected when any of these hold:

| Evidence | Example |
|---|---|
| Name or synonym mentioned | "clients" → customers |
| Column name mentioned | "category" → products |
| Real value mentioned | "Germany" → customers.country |
| Glossary term defined on it | "AOV" → orders |
| Similar meaning (embeddings) | "sent back" → orders.status, up to 2 tables, score ≥ 0.2 and ≥ 50% of the best |
| Needed to join the others | customers + products → orders, order_items |

If nothing matches, it falls back to the full schema. Few-shot examples are
also picked by meaning (`EmbeddingExampleSelector`).

Results on `knowledge/ecommerce/linking_eval.yaml` (25 questions phrased
unlike the schema): **25/25 covered, 3.3 of 6 tables per prompt** on average.
Exact matching alone covers 17/25. The thresholds were tuned on this same set,
so add held-out questions before trusting the numbers on a new database.
Slow test with the real model: `RUN_SLOW=1 .venv/bin/python -m pytest -m slow`.

## Step 4: Ambiguity handling (done)

Asks a clarifying question instead of guessing, but only when it matters.

```bash
export ANTHROPIC_API_KEY=...                       # needed from this step on
.venv/bin/python -m text2sql.ask                   # interactive session
.venv/bin/python -m text2sql.ask "What was revenue from India last quarter?"
```

```
Which revenue do you mean?
  [1] gross revenue: excludes cancelled orders
      revenue
      ---------
      812345.67
  [2] net revenue: also excludes returned orders
      ...
Choose 1-2:
```

How it decides:

| Trigger | What happens |
|---|---|
| Glossary term with several definitions ("revenue") | Model returns one interpretation + SQL per definition |
| Model spots another ambiguity ("best customers") | Same, with the model's own labels |
| Model guessed despite a glossary ambiguity | One query is generated per definition anyway |
| All interpretations return the same rows | Answers directly and states the assumption |
| User picked a definition earlier | Reused for the rest of the session ("sales" too, via synonyms) unless the question names another |

The model call uses `claude-opus-5-5` with structured outputs (the reply always
matches `RESPONSE_SCHEMA`), a cached system prompt, and server-side refusal
fallback. Interpretations are previewed through an **interim** read-only
executor (SELECT-only check + read-only connection); Phase 2 replaces it with
the full guardrail layer.

# Phase 2: SQL generation and safety

## Step 1: Validated SQL generation (done)

Every generated query (and every clarification option) is parsed with
`sqlglot` in the database's dialect and checked before it can run:

| Check | Catches | On failure |
|---|---|---|
| Syntax, single statement | `SELEC ...`, `SELECT 1; DROP ...` | Sent back to the model to fix |
| Every table exists | `FROM invoices` | Sent back to the model to fix |
| Every column exists (through aliases, CTEs, subqueries) | `o.customer_name`, `region` | Sent back to the model to fix |
| Unqualified column in several joined tables | `SELECT id FROM orders JOIN customers` | Sent back to the model to fix |
| Model's claimed tables/columns vs the real ones | Says `products`, reads `customers` | Shown as a warning |

Repairs reuse the same prompt plus a "previous answer was rejected" section
(at most 2). SQL that is still invalid is never executed. All 15 verified
examples pass with no false alarms. The structured response itself comes from
Claude's structured outputs (Phase 1, Step 4), so `instructor` isn't needed.

## Step 2: Guardrails (done)

Every query passes these rules before it runs (`src/text2sql/guardrails/`,
settings in `config/guardrails.yaml`):

| Rule | Blocks | Kind |
|---|---|---|
| read_only | Anything but SELECT/UNION/..., checked on **every node**, so `WITH d AS (DELETE ... RETURNING *) SELECT ...` is caught; also `SELECT INTO`, `FOR UPDATE`, PRAGMA, ATTACH, GRANT, COPY, SET, ... | hard |
| single_statement | `SELECT 1; DROP TABLE orders` | hard |
| blocked_function | `pg_sleep`, `pg_read_file`, `dblink`, `load_extension`, `randomblob`, `sleep`, ... | hard |
| subquery_depth | SELECTs nested deeper than 3 | retryable |
| table_access | Blocked tables, or tables outside an allow-list | retryable |
| sensitive_columns | Columns flagged as PII in Step 1 (including via `SELECT *`) | retryable |
| cartesian_join | Joins with no condition (equality or range) | retryable |
| row_limit | Adds `LIMIT 1000`, lowers bigger limits; skips single-row aggregates | rewrite |
| scan_estimate | Plans estimated to read > 1,000,000 rows (from EXPLAIN + Step 1 row counts) | retryable |

**Hard** violations are refused with no retry. **Retryable** ones go back to
the model with the reason ("reads personal data (customers.email); use ids or
aggregates"). Blocked and rewritten queries are appended to
`cache/logs/guardrails.jsonl`. The adversarial suite in
`tests/test_guardrails.py` has 59 attacks (all refused, none reach the
database) and 27 legitimate queries, including all 15 verified examples, that
must pass and run.

## Step 3: Sandbox (done)

Queries run where they can't do damage even if the guardrails miss something
(`src/text2sql/sandbox/`):

| Layer | SQLite | PostgreSQL |
|---|---|---|
| Read-only, always rolled back | read-only connection + `query_only` | `SET TRANSACTION READ ONLY`, rollback |
| Timeout (default 5s) | progress handler aborts the statement | `statement_timeout`, `lock_timeout` |
| The database refuses anything else | **authorizer** callback: only reads; PII columns, blocked tables and blocked functions denied | **SELECT-only role** with column-level grants that leave out PII |

A refusal from the database means an earlier layer let something through, so
it is logged as a `sandbox_refused` security event in the audit log.

Testing against PostgreSQL (Docker):

```bash
docker compose up -d --wait
cp .env.example .env && set -a && source .env && set +a   # dev-only values
.venv/bin/pip install -e ".[postgres]"
.venv/bin/python -m text2sql.sandbox.postgres setup --admin-url "$TEXT2SQL_PG_ADMIN_URL" --load-sample
.venv/bin/python -m text2sql.sandbox.postgres verify --db-url "$TEXT2SQL_PG_READER_URL"   # 18 checks
.venv/bin/python -m pytest tests/test_sandbox.py                                          # Postgres tests run too
.venv/bin/python -m text2sql.ask --db-url "$TEXT2SQL_PG_READER_URL"
```

## Step 4: Execution layer (done)

`create_service()` assembles every layer; each question returns one packaged
response (`src/text2sql/execution/`):

```python
service = create_service(sa.create_engine("sqlite:///data/ecommerce.db"))
r = service.ask("Top 10 products by units sold")
if r.status == "needs_clarification":
    r = service.choose(r.request_id, 0)
r.status, r.row_count, r.truncated, r.execution_ms, r.plan
df = r.dataframe()
```

| Field | Meaning |
|---|---|
| `status` | `ok`, `needs_clarification`, `cannot_answer`, `invalid_sql`, `blocked`, `timeout`, `denied`, `query_error`, `generation_failed` |
| `rows`, `columns`, `dataframe()` | Results, capped at the row limit |
| `row_count`, `truncated`, `row_limit` | `truncated` is true when the guardrails' LIMIT was reached, i.e. more rows exist |
| `execution_ms`, `total_ms` | Time in the database; question to response |
| `plan`, `estimated_scan_rows` | The EXPLAIN the guardrails already ran (not run twice) |
| `sql`, `generated_sql`, `rewrites`, `attempts` | What ran vs what the model wrote, and how |

Every question is appended to `cache/logs/queries.jsonl` (question, offered
tables, SQL, status, timings, row count, columns; never result data). A
clarification and its answer share one `request_id`. CLI extras:
`--plan` shows the query plan, `--csv out.csv` saves the rows.

# Phase 3: Hallucination detection (done)

Every answered question is checked and given a confidence band with reasons
(`src/text2sql/verification/`):

| Step | Check | Cost |
|---|---|---|
| 1. Back-translation | Claude describes the SQL **without seeing the question**; a second call compares that with the question aspect by aspect (measure, filters, grouping, time range, ordering/limit) | 2 calls |
| 2. Result sanity | Filter values that don't exist (`'delivered'` vs `'DELIVERED'`), dates outside the data, impossible counts, **join fan-out measured directly** (`COUNT(*)` vs `COUNT(DISTINCT key)` over the query's own joins), sums above the table total, values outside a column's range, NULL-heavy columns, duplicate rows, percentages outside 0–100 | free |
| 3. Multi-query | Complex queries (3+ tables, subqueries, windows, HAVING) are answered a second time with no examples and a different strategy, without seeing the first SQL; results are compared ignoring order, column names and rounding | 1 call |
| 4. Confidence | Weighted signals plus coverage (mentioned values filtered on, glossary conditions applied, a date filter for time phrases, `LIMIT N` for "top N"); serious findings cap the band | free |

Bands: **high** ≥ 0.80, **medium** ≥ 0.55, else **low**. A sanity error or
disagreeing second query caps at low; a back-translation mismatch or missing
coverage caps at medium; nothing independently verified caps at medium.

Measure it (asks before calling the API; ~100 calls for the full set):

```bash
.venv/bin/python -m text2sql.verification.evaluate
```

`knowledge/ecommerce/eval.yaml` holds 30 questions with verified answers
(case traps, gross vs net, joins, relative dates). The report gives accuracy,
accuracy per band (calibration), how many wrong answers stayed out of "high"
(detection) and how many right answers were marked low (false alarms).

# Phase 4: Query interface (done)

```bash
.venv/bin/pip install -e ".[web,embeddings]"
.venv/bin/python -m text2sql.api            # API on http://127.0.0.1:8000 (docs: /docs)
.venv/bin/python -m text2sql.api --demo     # no API key: replays the example library
.venv/bin/streamlit run ui/streamlit_app.py # UI on http://localhost:8501
```

## API (`src/text2sql/api/`)

| Endpoint | Does |
|---|---|
| `POST /v1/query` | Question → SQL, results, confidence (+ reasons and signals), guardrail rewrites and warnings; or a clarifying question |
| `POST /v1/query/{id}/choose` | Answer a clarifying question |
| `POST /v1/sql` | Run SQL the user edited, through the same validation, guardrails, sandbox and result checks |
| `GET /v1/schema` | Tables, columns, real category values (never personal data), relationships, glossary |
| `GET /v1/history` | The session's past questions and results, newest first |
| `POST /v1/feedback`, `GET /v1/feedback` | Mark an answer correct/incorrect (optionally with corrected SQL) |

Sessions are identified by a client-chosen `session_id`; clarification choices
and history are per session. There is no authentication: run it on localhost
or behind your own auth.

## UI (`ui/streamlit_app.py`)

Question box; SQL with syntax highlighting and an edit-and-rerun box; a
sortable results table with CSV download; the confidence band with problems,
supporting evidence and a per-signal breakdown; the second query's result when
it disagrees; the query plan; a history sidebar and a schema browser.

## Feedback flywheel, with a review gate (`src/text2sql/feedback/`)

| Feedback | Becomes | After |
|---|---|---|
| 👍 correct | candidate few-shot example (`examples.yaml`) | review |
| 👎 incorrect + corrected SQL | candidate test case (`eval.yaml`, gold = the fix) | review |
| 👎 incorrect, no correction | stays pending until someone writes the fix | - |

```bash
.venv/bin/python -m text2sql.feedback list
.venv/bin/python -m text2sql.feedback promote <id>   # re-validates, re-runs, then appends
.venv/bin/python -m text2sql.feedback reject <id> --reason "..."
```

Nothing changes the prompts automatically: a mistaken or malicious click could
otherwise teach the model wrong SQL. Promotion re-checks the SQL (validation,
guardrails, sandbox, non-empty result) and refuses duplicates.

# Phase 5: Evaluation suite and containers (done)

## Evaluation (`src/text2sql/evaluation/`)

```bash
.venv/bin/python -m text2sql.evaluation offline   # no API key: golden set, mutations, security
.venv/bin/python -m text2sql.evaluation live      # calls Claude (asks first): accuracy, behaviour
```

Both save to `docs/eval/` and rewrite the results block at the top of this
README and [docs/EVAL.md](EVAL.md), so the published numbers are always
the last measured ones; anything not measured says so.

| Suite | Measures | Needs |
|---|---|---|
| Golden set (`knowledge/ecommerce/eval.yaml`, 56 questions) | every verified answer runs; identical on SQLite and PostgreSQL (`--pg-url`) | - |
| Mutation suite | bugs injected into correct SQL (wrong value case, missing date/status filter, join fan-out, dropped LIMIT, wrong aggregate) that the checks flag; false alarms on the correct SQL | - |
| Security suite | 59 attacks on a database **copy**, fingerprinted before and after; the database layer alone with guardrails off; legitimate queries wrongly blocked | - |
| Live run | execution accuracy, normalised SQL exact match, asking on ambiguous and declining unanswerable questions, confidence calibration, prompt-injection questions | API key |

## Docker

```bash
docker compose up --build            # http://localhost:8501  (API: http://localhost:8000/docs)
docker build --target test .         # the test suite, inside the image (Python 3.12)
docker compose run --rm api evaluate offline
```

`setup` loads the sample data into PostgreSQL, creates the SELECT-only role
and verifies it; the API connects **as that role**. Without
`ANTHROPIC_API_KEY` the API runs in demo mode. The example and test SQL is
written for SQLite and translated to PostgreSQL automatically (sqlglot). The
embedding model is left out of the image (it would add ~2 GB of PyTorch), so
the container skips schema filtering.
