from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd

from utils.db import ensure_ewg_schema, exec_sql, get_engine, read_df

LOGGER = logging.getLogger("load_postgres")


def load_run_from_staging(*, run_id: str, database_url: Optional[str] = None, pipeline: str = "ewg_etl") -> dict[str, int]:
    engine = get_engine(database_url)
    ensure_ewg_schema(engine)

    products_out = read_df(
        engine,
        "SELECT category, company, product, url, ingredient_ids FROM stg_ewg_products_out WHERE run_id = %(run_id)s",
        {"run_id": run_id},
    )
    ingredients_dim = read_df(
        engine,
        "SELECT ingredient_id, ingredient, ingredient_url, functions, concerns FROM stg_ewg_ingredients_dim WHERE run_id = %(run_id)s",
        {"run_id": run_id},
    )
    product_ingredients = read_df(
        engine,
        "SELECT product_url, ingredient_id FROM stg_ewg_product_ingredients WHERE run_id = %(run_id)s",
        {"run_id": run_id},
    )

    if products_out.empty:
        LOGGER.info("Nothing to load for run_id=%s", run_id)
        return {"products": 0, "ingredients": 0, "junction": 0}

    # Normalize json columns
    if "ingredient_ids" in products_out.columns:
        products_out["ingredient_ids"] = products_out["ingredient_ids"].apply(lambda x: x if isinstance(x, list) else _as_list(x))

    for col in ["functions", "concerns"]:
        if col in ingredients_dim.columns:
            ingredients_dim[col] = ingredients_dim[col].apply(lambda x: x if isinstance(x, list) else _as_list(x))

    with engine.begin() as conn:
        # Upsert products
        products_out.to_sql("_tmp_products", con=conn, if_exists="replace", index=False)
        conn.exec_driver_sql(
            """
            INSERT INTO ewg_products (url, category, company, product, ingredient_ids, last_seen_at, last_run_id)
            SELECT url, category, company, product, ingredient_ids::jsonb, now(), %(run_id)s
            FROM _tmp_products
            ON CONFLICT (url) DO UPDATE SET
              category = EXCLUDED.category,
              company = EXCLUDED.company,
              product = EXCLUDED.product,
              ingredient_ids = EXCLUDED.ingredient_ids,
              last_seen_at = now(),
              last_run_id = %(run_id)s;
            DROP TABLE _tmp_products;
            """,
            {"run_id": run_id},
        )

        # Upsert ingredients dimension: merge lists by concatenation + distinct in SQL
        ingredients_dim.to_sql("_tmp_ingredients", con=conn, if_exists="replace", index=False)
        conn.exec_driver_sql(
            """
            INSERT INTO ewg_ingredients_dim (ingredient_id, ingredient, ingredient_url, functions, concerns, last_seen_at, last_run_id)
            SELECT ingredient_id, ingredient, ingredient_url, functions::jsonb, concerns::jsonb, now(), %(run_id)s
            FROM _tmp_ingredients
            ON CONFLICT (ingredient_id) DO UPDATE SET
              ingredient = COALESCE(EXCLUDED.ingredient, ewg_ingredients_dim.ingredient),
              ingredient_url = COALESCE(EXCLUDED.ingredient_url, ewg_ingredients_dim.ingredient_url),
              functions = (
                SELECT COALESCE(jsonb_agg(DISTINCT v), '[]'::jsonb)
                FROM (
                  SELECT jsonb_array_elements_text(ewg_ingredients_dim.functions) AS v
                  UNION ALL
                  SELECT jsonb_array_elements_text(EXCLUDED.functions) AS v
                ) t
              ),
              concerns = (
                SELECT COALESCE(jsonb_agg(DISTINCT v), '[]'::jsonb)
                FROM (
                  SELECT jsonb_array_elements_text(ewg_ingredients_dim.concerns) AS v
                  UNION ALL
                  SELECT jsonb_array_elements_text(EXCLUDED.concerns) AS v
                ) t
              ),
              last_seen_at = now(),
              last_run_id = %(run_id)s;
            DROP TABLE _tmp_ingredients;
            """,
            {"run_id": run_id},
        )

        # Upsert junction
        product_ingredients.to_sql("_tmp_pi", con=conn, if_exists="replace", index=False)
        conn.exec_driver_sql(
            """
            INSERT INTO ewg_product_ingredients (product_url, ingredient_id, last_seen_at, last_run_id)
            SELECT product_url, ingredient_id, now(), %(run_id)s
            FROM _tmp_pi
            ON CONFLICT (product_url, ingredient_id) DO UPDATE SET
              last_seen_at = now(),
              last_run_id = %(run_id)s;
            DROP TABLE _tmp_pi;
            """,
            {"run_id": run_id},
        )

        # Update checkpoints
        conn.exec_driver_sql(
            """
            INSERT INTO etl_checkpoints (pipeline, last_success_run_id, last_success_at)
            VALUES (%(pipeline)s, %(run_id)s, now())
            ON CONFLICT (pipeline) DO UPDATE SET
              last_success_run_id = EXCLUDED.last_success_run_id,
              last_success_at = EXCLUDED.last_success_at;
            """,
            {"pipeline": pipeline, "run_id": run_id},
        )

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
