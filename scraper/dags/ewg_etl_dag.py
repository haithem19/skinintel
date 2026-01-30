from __future__ import annotations

import os
import sys
from datetime import timedelta

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

# Ensure project root is on PYTHONPATH when DAG is parsed from dags/
DAG_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.environ.get("SCRAPER_PROJECT_ROOT") or os.path.abspath(os.path.join(DAG_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.extract_ewg import check_and_stage_discovery, extract_new_products_to_staging
from scripts.enrich_ingredients import enrich_ingredients_for_run
from scripts.transform_ewg import transform_run_to_staging
from scripts.load_postgres import load_run_from_staging


DEFAULT_ARGS = {
    "owner": "skinmate",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="ewg_etl",
    default_args=DEFAULT_ARGS,
    start_date=days_ago(1),
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["skinmate", "ewg"],
)
def ewg_etl_dag():
    # Use a single env var for scripts (works in Docker)
    # Prefer setting DATABASE_URL as a secret/env in your docker-compose.
    database_url = os.environ.get("DATABASE_URL")

    @task.short_circuit
    def has_new_products(run_id: str) -> bool:
        new_count = check_and_stage_discovery(run_id=run_id, database_url=database_url)
        return new_count > 0

    @task
    def extract(run_id: str) -> int:
        return extract_new_products_to_staging(run_id=run_id, database_url=database_url)

    @task
    def transform(run_id: str) -> dict:
        return transform_run_to_staging(run_id=run_id, database_url=database_url)

    @task
    def enrich_ingredients(run_id: str) -> dict:
        # Reads stg_ewg_ingredients_dim for this run_id and caches results in ewg_ingredients_enrichment.
        # If DEEPSEEK_API_KEY is not set, this task returns a skipped count and does not fail.
        return enrich_ingredients_for_run(run_id=run_id, database_url=database_url)

    @task
    def load(run_id: str) -> dict:
        return load_run_from_staging(run_id=run_id, database_url=database_url, pipeline="ewg_etl")

    # Use Airflow run_id string for uniqueness
    run_id = "{{ run_id }}"

    ok = has_new_products(run_id)
    extracted = extract(run_id)
    transformed = transform(run_id)
    enriched = enrich_ingredients(run_id)
    loaded = load(run_id)

    ok >> extracted >> transformed >> enriched >> loaded


ewg_etl_dag()
