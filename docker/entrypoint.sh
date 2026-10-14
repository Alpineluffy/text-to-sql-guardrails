#!/bin/sh
# Roles for the one image; see docker-compose.yml.
set -e
case "$1" in
  setup)
    # Load the sample data and (re)create the SELECT-only role. Idempotent.
    python -m text2sql.sandbox.postgres setup --admin-url "$TEXT2SQL_PG_ADMIN_URL" --load-sample
    python -m text2sql.sandbox.postgres verify --db-url "$TEXT2SQL_DB_URL"
    ;;
  api)
    if [ -n "$ANTHROPIC_API_KEY" ]; then
      exec python -m text2sql.api --host 0.0.0.0 --port 8000 --no-filter
    fi
    echo "ANTHROPIC_API_KEY is not set: starting in demo mode (answers replayed from the example library)."
    exec python -m text2sql.api --host 0.0.0.0 --port 8000 --no-filter --demo
    ;;
  ui)
    exec streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 \
      --server.headless true --browser.gatherUsageStats false
    ;;
  evaluate)
    shift
    exec python -m text2sql.evaluation "$@" --db-url "$TEXT2SQL_DB_URL" --no-filter --no-readme
    ;;
  *)
    exec "$@"
    ;;
esac
