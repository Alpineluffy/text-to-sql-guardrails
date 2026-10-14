"""Resolve relative date phrases up front so the model doesn't have to guess."""
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Tuple


def _quarter_bounds(year: int, quarter: int) -> Tuple[date, date]:
    start = date(year, 3 * quarter - 2, 1)
    end = date(year + 1, 1, 1) if quarter == 4 else date(year, 3 * quarter + 1, 1)
    return start, end - timedelta(days=1)


def date_context(today: date) -> List[str]:
    quarter = (today.month - 1) // 3 + 1
    q_start, q_end = _quarter_bounds(today.year, quarter)
    lq_year, lq = (today.year - 1, 4) if quarter == 1 else (today.year, quarter - 1)
    lq_start, lq_end = _quarter_bounds(lq_year, lq)
    month_start = today.replace(day=1)
    last_month_end = month_start - timedelta(days=1)

    return [
        f"Today: {today.isoformat()} ({today.strftime('%A')})",
        f"This month: {month_start} to {today}",
        f"Last month: {last_month_end.replace(day=1)} to {last_month_end}",
        f"This quarter (Q{quarter} {today.year}): {q_start} to {q_end}",
        f"Last quarter (Q{lq} {lq_year}): {lq_start} to {lq_end}",
        f"This year: {today.year}-01-01 to {today}",
        f"Last year: {today.year - 1}-01-01 to {today.year - 1}-12-31",
        f"Last 30 days: {today - timedelta(days=30)} to {today}",
    ]
