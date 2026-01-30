from __future__ import annotations

import json
import logging
import os
import sys
from typing import Iterable, Optional

import pandas as pd

# Ensure scripts directory is in path for relative imports
SCRIPTS_DIR = os.path.dirname(__file__)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from scrape_ewg import (
    DEFAULT_CATEGORIES,
    build_session,
    make_ingredient_id,
    scrape_ewg_category_products,
    scrape_ewg_product_ingredients_from_product_page,
    setup_logging,
)

from utils.db import (
    count_new_products,
    ensure_ewg_schema,
    fetch_known_ingredient_ids,
    get_engine,
    materialize_new_products,
    upsert_discovered_products,
    write_staging_ingredient_rows,
)

LOGGER = logging.getLogger("extract_ewg")


def _parse_categories(categories: Optional[Iterable[str]] = None) -> list[str]:
    cats = list(categories) if categories is not None else list(DEFAULT_CATEGORIES)
    cats = [c.strip() for c in cats if str(c).strip()]
    if not cats:
        raise ValueError("No categories configured")
    return cats


def discover_products_df(
    *,
    categories: Optional[Iterable[str]] = None,
    start_page: int = 1,
    end_page: Optional[int] = None,
    max_pages: Optional[int] = 1,
    max_products_per_category: Optional[int] = None,
    delay: float = 0.35,
    timeout: tuple[float, float] = (5.0, 30.0),
) -> pd.DataFrame:
    setup_logging()
    sess = build_session()

    rows: list[dict] = []
    for cat in _parse_categories(categories):
        LOGGER.info("Discovering products category=%s", cat)
        products = scrape_ewg_category_products(
            category=cat,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
            max_products=max_products_per_category,
            delay=delay,
            timeout=timeout,
            session=sess,
        )
        rows.extend([p.__dict__ for p in products])

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["category", "company", "product", "url"])
    return df


def check_and_stage_discovery(
    *,
    run_id: str,
    database_url: Optional[str] = None,
    categories: Optional[Iterable[str]] = None,
) -> int:
    """Lightweight check step.

    - Scrapes ONLY category listings.
    - Stores them in Postgres staging.
    - Returns number of NEW product URLs not yet present in final table.
    """

    engine = get_engine(database_url)
    ensure_ewg_schema(engine)

    discovered = discover_products_df(categories=categories)
    upsert_discovered_products(engine, run_id=run_id, products_df=discovered)

    new_count = count_new_products(engine, run_id=run_id)
    LOGGER.info("New products detected: %s", new_count)
    return new_count


def extract_new_products_to_staging(
    *,
    run_id: str,
    database_url: Optional[str] = None,
    categories: Optional[Iterable[str]] = None,
    max_ingredients_per_product: Optional[int] = None,
    delay: float = 0.35,
    timeout: tuple[float, float] = (5.0, 30.0),
) -> int:
    """Heavy extract step.

    - Uses the discovery staging table for this run.
    - Only scrapes products whose URL is not already in ewg_products.
    - Writes ingredient rows into Postgres staging.
    - Only keeps functions/concerns for ingredients not already in ewg_ingredients_dim.
    """

    setup_logging()
    engine = get_engine(database_url)
    ensure_ewg_schema(engine)

    materialize_new_products(engine, run_id=run_id)

    known_ingredient_ids = fetch_known_ingredient_ids(engine)
    sess = build_session()

    # Read new products directly from staging
    new_products = pd.read_sql_query(
        "SELECT category, company, product, url FROM stg_ewg_products_new WHERE run_id = %(run_id)s ORDER BY url",
        con=engine,
        params={"run_id": run_id},
    )

    if new_products.empty:
        LOGGER.info("No new products to extract for run_id=%s", run_id)
        return 0

    ingredient_rows: list[dict] = []

    for idx, row in enumerate(new_products.to_dict(orient="records"), start=1):
        url = str(row.get("url"))
        LOGGER.info("Scraping product %s/%s | %s", idx, len(new_products), url)

        ing = scrape_ewg_product_ingredients_from_product_page(
            url,
            max_ingredients=max_ingredients_per_product,
            timeout=timeout,
            session=sess,
        )

        for ing_row in ing:
            ingredient = ing_row.get("ingredient")
            ingredient_url = ing_row.get("ingredient_url")
            ingredient_id = make_ingredient_id(ingredient, ingredient_url)
            is_new = ingredient_id not in known_ingredient_ids

            ingredient_rows.append(
                {
                    "category": row.get("category"),
                    "company": row.get("company"),
                    "product": row.get("product"),
                    "product_url": url,
                    "ingredient_id": ingredient_id,
                    "ingredient": ingredient,
                    "ingredient_url": ingredient_url,
                    "functions": ing_row.get("functions") if is_new else [],
                    "concerns": ing_row.get("concerns") if is_new else [],
                }
            )

            if is_new:
                known_ingredient_ids.add(ingredient_id)

        if delay:
            import time
            import random

            time.sleep(delay + random.uniform(0, min(0.25, delay)))

    df = pd.DataFrame(ingredient_rows)
    # Ensure functions/concerns are JSON arrays
    if not df.empty:
        for col in ["functions", "concerns"]:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

    write_staging_ingredient_rows(engine, run_id=run_id, ingredient_rows_df=df)
    return len(df)
