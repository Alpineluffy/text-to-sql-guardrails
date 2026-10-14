"""Query execution lives in text2sql.sandbox; these names are kept for callers."""
from ..sandbox.executor import (  # noqa: F401
    QueryResult, SandboxedExecutor, SandboxError, UnsafeQueryError, check_read_only,
)

# The Phase 1 name; it is now the full sandbox.
ReadOnlyExecutor = SandboxedExecutor
