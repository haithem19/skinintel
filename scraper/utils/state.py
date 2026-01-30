from __future__ import annotations

import os
from datetime import datetime, timezone


def airflow_run_id(context: dict | None = None) -> str:
    """Return a stable run_id string for a DAG run.

    In Airflow this can be context['run_id']; we fall back to execution date.
    """

    if context:
        rid = context.get("run_id")
        if rid:
            return str(rid)
        dt = context.get("logical_date") or context.get("execution_date")
        if dt:
            try:
                return str(dt)
            except Exception:
                pass
    # local fallback
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def env_pipeline_name(default: str = "ewg_etl") -> str:
    return os.environ.get("PIPELINE_NAME", default)
