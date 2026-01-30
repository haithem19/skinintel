from __future__ import annotations

import os
import json
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

import pandas as pd

import psycopg2


@dataclass(frozen=True)
class PostgresConfig:
    """DB config resolved from env vars.

    Prefer setting DATABASE_URL.
    In Airflow, you can also set AIRFLOW_CONN_SKINMATE_POSTGRES and pass it via PostgresHook.
    """

    database_url: str


def load_postgres_config() -> PostgresConfig:
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return PostgresConfig(database_url=database_url)

    # Common Docker/Airflow env vars
    # (matches the defaults you provided)
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    if db_host or db_port or db_name or db_user or db_password:
        host = db_host or "postgres"
        port = str(db_port or "5432")
        user = db_user or "airflow"
        password = db_password or "airflow"
        db = db_name or "airflow"
        return PostgresConfig(database_url=f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

    # Fallback: build from discrete env vars (useful outside Airflow)
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    user = os.environ.get("PGUSER", "postgres")
    password = os.environ.get("PGPASSWORD", "postgres")
    db = os.environ.get("PGDATABASE", "postgres")
    return PostgresConfig(database_url=f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")


def get_engine(database_url: Optional[str] = None):
    """Create a SQLAlchemy engine.

    This stays out of Airflow specifics so scripts can run locally too.
    """

    from sqlalchemy import create_engine

    url = database_url or load_postgres_config().database_url
    return create_engine(url, pool_pre_ping=True)


def _parse_database_url(database_url: str) -> dict[str, Any]:
    """Parse DATABASE_URL into kwargs usable by psycopg2.connect()."""

    u = urlparse(database_url)
    if u.scheme not in {"postgres", "postgresql"}:
        # Accept SQLAlchemy style URL too (postgresql+psycopg2)
        if u.scheme.startswith("postgresql"):
            pass
        else:
            raise ValueError("Unsupported DATABASE_URL scheme for psycopg2")

    dbname = (u.path or "").lstrip("/")
    return {
        "host": u.hostname or "postgres",
        "port": int(u.port or 5432),
        "database": dbname or os.getenv("DB_NAME", "airflow"),
        "user": u.username or os.getenv("DB_USER", "airflow"),
        "password": u.password or os.getenv("DB_PASSWORD", "airflow"),
    }


def get_db_connection():
    """Creates a PostgreSQL connection using psycopg2.

    Env precedence:
    - DATABASE_URL (postgresql://...) if set
    - DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD
    - Fallback defaults (postgres/5432/airflow/airflow/airflow)
    """

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        kwargs = _parse_database_url(database_url)
        return psycopg2.connect(**kwargs)

    return psycopg2.connect(
        host=os.getenv("DB_HOST", "postgres"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "airflow"),
        user=os.getenv("DB_USER", "airflow"),
        password=os.getenv("DB_PASSWORD", "airflow"),
    )


@contextmanager
def get_db_cursor(commit: bool = True):
    """Context manager for psycopg2 database operations."""

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def exec_sql(engine, sql: str, params: Optional[dict[str, Any]] = None) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(sql, params or {})


def fetch_scalar(engine, sql: str, params: Optional[dict[str, Any]] = None):
    with engine.begin() as conn:
        res = conn.exec_driver_sql(sql, params or {})
        row = res.first()
        return None if row is None else row[0]


def read_df(engine, sql: str, params: Optional[dict[str, Any]] = None) -> pd.DataFrame:
    return pd.read_sql_query(sql, con=engine, params=params)


def write_df(engine, df: pd.DataFrame, table: str, *, if_exists: str = "append") -> None:
    # pandas will create tables if they don't exist; we prefer explicit DDL in ensure_*.
    df.to_sql(table, con=engine, if_exists=if_exists, index=False, method="multi", chunksize=1000)


def ensure_ewg_schema(engine) -> None:
    """Create required tables for the EWG ETL if missing.

    Staging tables are run-scoped via run_id.
    Final tables are idempotent and upserted.
    """

    exec_sql(
        engine,
        """
        CREATE TABLE IF NOT EXISTS etl_runs (
          pipeline TEXT NOT NULL,
          run_id TEXT NOT NULL,
          started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          finished_at TIMESTAMPTZ NULL,
          status TEXT NULL,
          details JSONB NULL,
          PRIMARY KEY (pipeline, run_id)
        );

        CREATE TABLE IF NOT EXISTS etl_checkpoints (
          pipeline TEXT PRIMARY KEY,
          last_success_run_id TEXT NULL,
          last_success_at TIMESTAMPTZ NULL
        );

        CREATE TABLE IF NOT EXISTS stg_ewg_products_discovered (
          run_id TEXT NOT NULL,
          category TEXT NULL,
          company TEXT NULL,
          product TEXT NULL,
          url TEXT NOT NULL,
          discovered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          PRIMARY KEY (run_id, url)
        );

        CREATE TABLE IF NOT EXISTS stg_ewg_products_new (
          run_id TEXT NOT NULL,
          category TEXT NULL,
          company TEXT NULL,
          product TEXT NULL,
          url TEXT NOT NULL,
          PRIMARY KEY (run_id, url)
        );

        CREATE TABLE IF NOT EXISTS stg_ewg_ingredient_rows (
          run_id TEXT NOT NULL,
          category TEXT NULL,
          company TEXT NULL,
          product TEXT NULL,
          product_url TEXT NOT NULL,
          ingredient_id TEXT NOT NULL,
          ingredient TEXT NULL,
          ingredient_url TEXT NULL,
          functions JSONB NULL,
          concerns JSONB NULL
        );

        CREATE TABLE IF NOT EXISTS stg_ewg_products_out (
          run_id TEXT NOT NULL,
          category TEXT NULL,
          company TEXT NULL,
          product TEXT NULL,
          url TEXT NOT NULL,
          ingredient_ids JSONB NOT NULL,
          PRIMARY KEY (run_id, url)
        );

        CREATE TABLE IF NOT EXISTS stg_ewg_ingredients_dim (
          run_id TEXT NOT NULL,
          ingredient_id TEXT NOT NULL,
          ingredient TEXT NULL,
          ingredient_url TEXT NULL,
          functions JSONB NOT NULL,
          concerns JSONB NOT NULL,
          PRIMARY KEY (run_id, ingredient_id)
        );

        CREATE TABLE IF NOT EXISTS stg_ewg_product_ingredients (
          run_id TEXT NOT NULL,
          product_url TEXT NOT NULL,
          ingredient_id TEXT NOT NULL,
          PRIMARY KEY (run_id, product_url, ingredient_id)
        );

        CREATE TABLE IF NOT EXISTS ewg_products (
          url TEXT PRIMARY KEY,
          category TEXT NULL,
          company TEXT NULL,
          product TEXT NULL,
          ingredient_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
          first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          last_run_id TEXT NULL
        );

        CREATE TABLE IF NOT EXISTS ewg_ingredients_dim (
          ingredient_id TEXT PRIMARY KEY,
          ingredient TEXT NULL,
          ingredient_url TEXT NULL,
          functions JSONB NOT NULL DEFAULT '[]'::jsonb,
          concerns JSONB NOT NULL DEFAULT '[]'::jsonb,
                    skin_type_compatibility TEXT[] NOT NULL DEFAULT '{}'::text[],
                    interactions TEXT[] NOT NULL DEFAULT '{}'::text[],
                    recommendation_time TEXT[] NOT NULL DEFAULT '{}'::text[],
          first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          last_run_id TEXT NULL
        );

                -- Backward-compatible migration for existing databases.
                -- If prior versions created JSONB enrichment columns, rename them to avoid name/type conflicts.
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'ewg_ingredients_dim' AND column_name = 'skin_types'
                    ) AND NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'ewg_ingredients_dim' AND column_name = 'skin_types_json'
                    ) THEN
                        ALTER TABLE ewg_ingredients_dim RENAME COLUMN skin_types TO skin_types_json;
                    END IF;
                END $$;

                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'ewg_ingredients_dim' AND column_name = 'interactions'
                            AND data_type = 'jsonb'
                    ) AND NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'ewg_ingredients_dim' AND column_name = 'interactions_json'
                    ) THEN
                        ALTER TABLE ewg_ingredients_dim RENAME COLUMN interactions TO interactions_json;
                    END IF;
                END $$;

                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'ewg_ingredients_dim' AND column_name = 'recommendation_time'
                            AND data_type = 'jsonb'
                    ) AND NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'ewg_ingredients_dim' AND column_name = 'recommendation_time_json'
                    ) THEN
                        ALTER TABLE ewg_ingredients_dim RENAME COLUMN recommendation_time TO recommendation_time_json;
                    END IF;
                END $$;

                ALTER TABLE ewg_ingredients_dim ADD COLUMN IF NOT EXISTS skin_type_compatibility TEXT[] NOT NULL DEFAULT '{}'::text[];
                ALTER TABLE ewg_ingredients_dim ADD COLUMN IF NOT EXISTS interactions TEXT[] NOT NULL DEFAULT '{}'::text[];
                ALTER TABLE ewg_ingredients_dim ADD COLUMN IF NOT EXISTS recommendation_time TEXT[] NOT NULL DEFAULT '{}'::text[];

        CREATE TABLE IF NOT EXISTS ewg_product_ingredients (
          product_url TEXT NOT NULL,
          ingredient_id TEXT NOT NULL,
          first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          last_run_id TEXT NULL,
          PRIMARY KEY (product_url, ingredient_id)
        );
        """,
    )


def upsert_discovered_products(engine, *, run_id: str, products_df: pd.DataFrame) -> None:
    if products_df.empty:
        return

    df = products_df.copy()
    df["run_id"] = run_id
    cols = ["run_id", "category", "company", "product", "url"]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    # Insert by dataframe then rely on ON CONFLICT to dedupe.
    with engine.begin() as conn:
        tmp = "_tmp_discovered"
        df[cols].to_sql(tmp, con=conn, if_exists="replace", index=False)
        conn.exec_driver_sql(
            f"""
            INSERT INTO stg_ewg_products_discovered (run_id, category, company, product, url)
            SELECT run_id, category, company, product, url FROM {tmp}
            ON CONFLICT (run_id, url) DO UPDATE SET
              category = EXCLUDED.category,
              company = EXCLUDED.company,
              product = EXCLUDED.product;
            DROP TABLE {tmp};
            """
        )


def count_new_products(engine, *, run_id: str) -> int:
    sql = """
    SELECT COUNT(*)
    FROM stg_ewg_products_discovered d
    LEFT JOIN ewg_products p ON p.url = d.url
    WHERE d.run_id = %(run_id)s
      AND p.url IS NULL;
    """
    v = fetch_scalar(engine, sql, {"run_id": run_id})
    return int(v or 0)


def materialize_new_products(engine, *, run_id: str) -> int:
    """Copy discovered products for this run that are not in ewg_products into stg_ewg_products_new."""

    with engine.begin() as conn:
        res = conn.exec_driver_sql(
            """
            INSERT INTO stg_ewg_products_new (run_id, category, company, product, url)
            SELECT d.run_id, d.category, d.company, d.product, d.url
            FROM stg_ewg_products_discovered d
            LEFT JOIN ewg_products p ON p.url = d.url
            WHERE d.run_id = %(run_id)s
              AND p.url IS NULL
            ON CONFLICT (run_id, url) DO NOTHING;
            """,
            {"run_id": run_id},
        )
        return res.rowcount or 0


def fetch_new_product_urls(engine, *, run_id: str) -> list[str]:
    df = read_df(
        engine,
        "SELECT url FROM stg_ewg_products_new WHERE run_id = %(run_id)s ORDER BY url",
        {"run_id": run_id},
    )
    return [str(x) for x in df["url"].tolist()] if not df.empty else []


def fetch_known_ingredient_ids(engine) -> set[str]:
    df = read_df(engine, "SELECT ingredient_id FROM ewg_ingredients_dim")
    return {str(x) for x in df["ingredient_id"].dropna().tolist()} if not df.empty else set()


def write_staging_ingredient_rows(engine, *, run_id: str, ingredient_rows_df: pd.DataFrame) -> None:
    if ingredient_rows_df.empty:
        return

    df = ingredient_rows_df.copy()
    df["run_id"] = run_id

    # Ensure jsonb columns are actual JSON strings/objects
    for col in ["functions", "concerns"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, (dict, list)) or x is None else _safe_json(x))

    cols = [
        "run_id",
        "category",
        "company",
        "product",
        "product_url",
        "ingredient_id",
        "ingredient",
        "ingredient_url",
        "functions",
        "concerns",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    df = df[cols]
    write_df(engine, df, "stg_ewg_ingredient_rows", if_exists="append")


def clear_run_staging(engine, *, run_id: str) -> None:
    exec_sql(
        engine,
        """
        DELETE FROM stg_ewg_product_ingredients WHERE run_id = %(run_id)s;
        DELETE FROM stg_ewg_ingredients_dim WHERE run_id = %(run_id)s;
        DELETE FROM stg_ewg_products_out WHERE run_id = %(run_id)s;
        DELETE FROM stg_ewg_ingredient_rows WHERE run_id = %(run_id)s;
        DELETE FROM stg_ewg_products_new WHERE run_id = %(run_id)s;
        DELETE FROM stg_ewg_products_discovered WHERE run_id = %(run_id)s;
        """,
        {"run_id": run_id},
    )


def _safe_json(x: Any):
    try:
        return json.loads(x) if isinstance(x, str) else x
    except Exception:
        return []
