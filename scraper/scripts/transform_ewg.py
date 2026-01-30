from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd

from scrape_ewg import build_sql_like_tables
from utils.db import ensure_ewg_schema, exec_sql, get_engine, read_df, write_df

LOGGER = logging.getLogger("transform_ewg")


def transform_run_to_staging(*, run_id: str, database_url: Optional[str] = None) -> dict[str, int]:
    """Transform staging raw rows into staging transformed tables."""

    engine = get_engine(database_url)
    ensure_ewg_schema(engine)

    products_df = read_df(
        engine,
        "SELECT category, company, product, url FROM stg_ewg_products_new WHERE run_id = %(run_id)s",
        {"run_id": run_id},
    )
    ingredients_df = read_df(
        engine,
        """
        SELECT category, company, product, product_url, ingredient, ingredient_url, ingredient_id, functions, concerns
        FROM stg_ewg_ingredient_rows
        WHERE run_id = %(run_id)s
        """,
        {"run_id": run_id},
    )

    if products_df.empty:
        LOGGER.info("No staging products for run_id=%s", run_id)
        return {"products": 0, "ingredients": 0, "junction": 0}

    # Normalize JSONB columns: psycopg2 may return as strings depending on driver
    for col in ["functions", "concerns"]:
        if col in ingredients_df.columns:
            ingredients_df[col] = ingredients_df[col].apply(lambda x: x if isinstance(x, list) else _as_list(x))

    products_out, ingredients_dim, product_ingredients = build_sql_like_tables(
        products_df=products_df,
        ingredients_df=ingredients_df,
    )

    # Convert ingredient_ids JSON-string column into a real list for jsonb storage
    if "ingredient_ids" in products_out.columns:
        products_out["ingredient_ids"] = products_out["ingredient_ids"].apply(lambda s: json.loads(s) if isinstance(s, str) else s)

    products_out["run_id"] = run_id
    ingredients_dim["run_id"] = run_id
    product_ingredients["run_id"] = run_id

    # Replace previous staging outputs for this run
    exec_sql(engine, "DELETE FROM stg_ewg_products_out WHERE run_id = %(run_id)s", {"run_id": run_id})
    exec_sql(engine, "DELETE FROM stg_ewg_ingredients_dim WHERE run_id = %(run_id)s", {"run_id": run_id})
    exec_sql(engine, "DELETE FROM stg_ewg_product_ingredients WHERE run_id = %(run_id)s", {"run_id": run_id})

    write_df(engine, products_out[["run_id", "category", "company", "product", "url", "ingredient_ids"]], "stg_ewg_products_out")

    # Ensure lists are stored as JSON arrays
    for col in ["functions", "concerns"]:
        if col in ingredients_dim.columns:
            ingredients_dim[col] = ingredients_dim[col].apply(lambda x: x if isinstance(x, list) else [])

    write_df(
        engine,
        ingredients_dim[["run_id", "ingredient_id", "ingredient", "ingredient_url", "functions", "concerns"]],
        "stg_ewg_ingredients_dim",
    )
    write_df(engine, product_ingredients[["run_id", "product_url", "ingredient_id"]], "stg_ewg_product_ingredients")

    return {
        "products": int(len(products_out)),
        "ingredients": int(len(ingredients_dim)),
        "junction": int(len(product_ingredients)),
    }


def _as_list(x):
    if isinstance(x, list):
        return x
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() in {"nan", "none"}:
            return []
        try:
            v = json.loads(s)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []
