from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import requests

from utils.db import ensure_ewg_schema, get_engine, read_df

LOGGER = logging.getLogger("enrich_ingredients")


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    timeout_s: float = 45.0
    prompt_version: str = "v1"


SYSTEM_PROMPT = (
    "You are a cosmetic ingredient analyst. "
    "Return ONLY valid JSON. No markdown. No extra keys. "
    "If unknown, use empty arrays." 
)


def _build_user_prompt(*, ingredient: str, functions: list[str], concerns: list[str]) -> str:
    payload: dict[str, Any] = {
        "task": "enrich_cosmetic_ingredient",
        "ingredient": ingredient,
        "output_schema": {
            "skin_type_compatibility": [
                "oily",
                "dry",
                "combination",
                "normal",
                "sensitive",
                "acne_prone",
                "mature",
                "all"
            ],
            "interactions": ["<list of ingredients that should NOT be used with this ingredient>"],
            "recommendation_time": ["morning", "evening"],
        },
        "rules": [
            "Return ONLY these 3 keys: skin_type_compatibility, interactions, recommendation_time.",
            "All values must be arrays of lowercase snake_case strings.",
            "skin_type_compatibility: which skin types benefit from or tolerate this ingredient.",
            "interactions: list ONLY ingredients that have NEGATIVE interactions with this ingredient (should NOT be used together due to irritation, reduced efficacy, or instability). Examples: retinol + aha, vitamin_c + benzoyl_peroxide, niacinamide + vitamin_c (disputed), retinol + benzoyl_peroxide. Return empty array if no known negative interactions.",
            "recommendation_time: morning, evening, or both.",
            "Keep arrays reasonably sized (<=8 items).",
            "Avoid medical claims; focus on cosmetic compatibility/irritation.",
        ],
    }

    # Only include known lists if we actually have them (saves tokens).
    if functions:
        payload["known_functions"] = functions[:10]
    if concerns:
        payload["known_concerns"] = concerns[:10]

    return json.dumps(payload, ensure_ascii=False)


def _extract_content(resp_json: dict[str, Any]) -> str:
    try:
        return str(resp_json["choices"][0]["message"]["content"])
    except Exception as e:
        raise ValueError(f"Unexpected DeepSeek response shape: {resp_json}") from e


def _norm_token(s: str) -> str:
    v = (s or "").strip().lower()
    v = v.replace("-", "_").replace(" ", "_")
    while "__" in v:
        v = v.replace("__", "_")
    return v.strip("_")


def _normalize_str_list(x: Any, *, max_len: int) -> list[str]:
    if not isinstance(x, list):
        return []
    out: list[str] = []
    for item in x:
        tok = _norm_token(str(item))
        if not tok:
            continue
        if tok not in out:
            out.append(tok)
        if len(out) >= max_len:
            break
    return out


def _normalize_enrichment(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {
            "skin_type_compatibility": [],
            "interactions": [],
            "recommendation_time": [],
        }

    skin_types = _normalize_str_list(obj.get("skin_type_compatibility"), max_len=8)
    # interactions: accept any ingredient names (no whitelist), allow more items
    interactions = _normalize_str_list(obj.get("interactions"), max_len=12)
    recommendation_time = _normalize_str_list(obj.get("recommendation_time"), max_len=2)

    allowed_skin = {"oily", "dry", "combination", "normal", "sensitive", "acne_prone", "mature", "all"}
    skin_types = [s for s in skin_types if s in allowed_skin]

    allowed_times = {"morning", "evening"}
    recommendation_time = [t for t in recommendation_time if t in allowed_times]

    return {
        "skin_type_compatibility": skin_types,
        "interactions": interactions,
        "recommendation_time": recommendation_time,
    }


def _to_pg_text_array_literal(values: list[str]) -> str:
    # Postgres array literal: {"a","b"}. Escape backslashes and quotes.
    if not values:
        return "{}"
    escaped = []
    for v in values:
        s = str(v)
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        escaped.append(f'"{s}"')
    return "{" + ",".join(escaped) + "}"


def _deepseek_chat_json(*, cfg: DeepSeekConfig, user_prompt: str) -> dict[str, Any]:
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": cfg.model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    resp = requests.post(url, headers=headers, json=body, timeout=cfg.timeout_s)
    resp.raise_for_status()
    return resp.json()


def enrich_ingredients_for_run(
    *,
    run_id: str,
    database_url: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    deepseek_base_url: Optional[str] = None,
    deepseek_model: Optional[str] = None,
) -> dict[str, int]:
    """Enrich deduplicated ingredient dimension for a run.

    Reads:
      - stg_ewg_ingredients_dim (for run_id)
            - ewg_ingredients_dim (enrichment cache via recommendation_time)

    Writes:
            - ewg_ingredients_dim (upsert of skin_types/interactions/recommendation_time)

    Notes:
      - Calls DeepSeek at most once per missing ingredient_id.
      - If no API key is configured, does nothing and returns skipped counts.
    """

    engine = get_engine(database_url)
    ensure_ewg_schema(engine)

    api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        df = read_df(
            engine,
            "SELECT ingredient_id FROM stg_ewg_ingredients_dim WHERE run_id = %(run_id)s",
            {"run_id": run_id},
        )
        return {"enriched": 0, "skipped_no_key": int(len(df))}

    cfg = DeepSeekConfig(
        api_key=api_key,
        base_url=deepseek_base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        model=deepseek_model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    )

    run_ing = read_df(
        engine,
        """
        SELECT ingredient_id, ingredient, ingredient_url, functions, concerns
        FROM stg_ewg_ingredients_dim
        WHERE run_id = %(run_id)s
        ORDER BY ingredient_id
        """,
        {"run_id": run_id},
    )
    if run_ing.empty:
        return {"enriched": 0, "skipped_cached": 0}

        # Determine which ingredient_ids already have enrichment.
        # We treat recommendation_time (TEXT[]) non-empty as "already enriched" to avoid repeated calls.
    existing = read_df(
        engine,
        """
        SELECT ingredient_id
        FROM ewg_ingredients_dim
                WHERE recommendation_time IS NOT NULL
                    AND cardinality(recommendation_time) > 0
        """,
    )
    existing_ids = set(str(x) for x in existing["ingredient_id"].dropna().tolist()) if not existing.empty else set()

    to_enrich = run_ing[~run_ing["ingredient_id"].astype(str).isin(existing_ids)].copy()
    if to_enrich.empty:
        return {"enriched": 0, "skipped_cached": int(len(run_ing))}

    enriched_rows: list[dict[str, Any]] = []

    for r in to_enrich.to_dict(orient="records"):
        ingredient_id = str(r.get("ingredient_id") or "").strip()
        ingredient = str(r.get("ingredient") or "").strip()
        ingredient_url = r.get("ingredient_url")
        ingredient_url = str(ingredient_url).strip() if ingredient_url is not None and str(ingredient_url).strip() else None
        if not ingredient_id:
            continue
        if not ingredient:
            ingredient = ingredient_id

        functions = r.get("functions") if isinstance(r.get("functions"), list) else []
        concerns = r.get("concerns") if isinstance(r.get("concerns"), list) else []

        user_prompt = _build_user_prompt(ingredient=ingredient, functions=functions, concerns=concerns)

        try:
            resp_json = _deepseek_chat_json(cfg=cfg, user_prompt=user_prompt)
            content = _extract_content(resp_json)
            parsed = json.loads(content)
            norm = _normalize_enrichment(parsed)
            enriched_rows.append(
                {
                    "ingredient_id": ingredient_id,
                    "ingredient": ingredient,
                    "ingredient_url": ingredient_url,
                    "skin_type_compatibility": _to_pg_text_array_literal(norm["skin_type_compatibility"]),
                    "interactions": _to_pg_text_array_literal(norm["interactions"]),
                    "recommendation_time": _to_pg_text_array_literal(norm["recommendation_time"]),
                }
            )
        except Exception as e:
            LOGGER.warning("Enrichment failed ingredient_id=%s ingredient=%s err=%s", ingredient_id, ingredient, e)

    if not enriched_rows:
        return {"enriched": 0, "skipped_cached": int(len(run_ing))}

    out = pd.DataFrame(enriched_rows)

    with engine.begin() as conn:
        out.to_sql("_tmp_ing_enrich", con=conn, if_exists="replace", index=False)
        conn.exec_driver_sql(
            """
                        INSERT INTO ewg_ingredients_dim (
              ingredient_id,
                            ingredient,
                            ingredient_url,
                            skin_type_compatibility,
              interactions,
              recommendation_time,
                            last_seen_at,
                            last_run_id
            )
            SELECT
              ingredient_id,
                            ingredient,
                            ingredient_url,
                            skin_type_compatibility::text[],
                            interactions::text[],
                            recommendation_time::text[],
                            now(),
                            %(run_id)s
            FROM _tmp_ing_enrich
            ON CONFLICT (ingredient_id) DO UPDATE SET
                            ingredient = COALESCE(EXCLUDED.ingredient, ewg_ingredients_dim.ingredient),
                            ingredient_url = COALESCE(EXCLUDED.ingredient_url, ewg_ingredients_dim.ingredient_url),
                            skin_type_compatibility = EXCLUDED.skin_type_compatibility,
                            interactions = EXCLUDED.interactions,
                            recommendation_time = EXCLUDED.recommendation_time,
                            last_seen_at = now(),
                            last_run_id = %(run_id)s;
            DROP TABLE _tmp_ing_enrich;
                        """,
                        {"run_id": run_id},
        )

    return {"enriched": int(len(out)), "skipped_cached": int(len(run_ing) - len(to_enrich))}
