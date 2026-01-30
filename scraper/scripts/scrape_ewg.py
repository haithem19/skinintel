from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path
from dataclasses import dataclass
import csv
from typing import Iterable, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.ewg.org"

# Edit this list for your one-time snapshot run.
# Values must match the category segment used in the EWG URL.
DEFAULT_CATEGORIES: list[str] = [
    "Anti-Aging",
    "Around-Eye_Cream",
    "BB_Cream",
    "CC_Cream",
    "Facial Cleanser",
    "Facial_Moisturizer__Treatment",
    "Makeup_remover",
    "Mask",
    "Oil_Controller",
    "Pore_Strips",
    "Serums_&_Essences",
    "Skin_Fading__Lightener",
    "Toners__Astringents",
]

LOGGER = logging.getLogger("scrape_ewg")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _append_csv_row(path: Path, fieldnames: list[str], row: dict) -> None:
    _ensure_dir(path.parent)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fieldnames})


def _as_list(x) -> list:
    """Best-effort: parse a value into a Python list.

    Used for reading CSV-cached JSON columns that may be stored as JSON strings.
    """

    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() in {"nan", "none"}:
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
        except Exception:
            return []
    return []


def _union_str_lists(values: Iterable) -> list[str]:
    acc: list[str] = []
    seen: set[str] = set()
    for v in values:
        for item in _as_list(v):
            t = _norm(str(item))
            if not t:
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            acc.append(t)
    return acc


def _load_ingredients_dim_cache(path: Path) -> pd.DataFrame:
    """Load cached unique ingredient details (if present)."""
    if not path.exists():
        return pd.DataFrame(columns=["ingredient_id", "ingredient", "ingredient_url", "functions", "concerns"])
    df = pd.read_csv(path)
    # Normalize expected columns
    for col in ["ingredient_id", "ingredient", "ingredient_url", "functions", "concerns"]:
        if col not in df.columns:
            df[col] = None
    if "ingredient_id" in df.columns:
        df["ingredient_id"] = df["ingredient_id"].where(pd.notna(df["ingredient_id"]), None)
    if "ingredient" in df.columns:
        df["ingredient"] = df["ingredient"].where(pd.notna(df["ingredient"]), "").astype(str)
    if "ingredient_url" in df.columns:
        df["ingredient_url"] = df["ingredient_url"].where(pd.notna(df["ingredient_url"]), None)
    return df[["ingredient_id", "ingredient", "ingredient_url", "functions", "concerns"]]


def _merge_ingredients_dim(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        base = new.copy() if new is not None else pd.DataFrame()
    elif new is None or new.empty:
        base = existing.copy()
    else:
        base = pd.concat([existing, new], ignore_index=True)

    if base.empty:
        return pd.DataFrame(columns=["ingredient_id", "ingredient", "ingredient_url", "functions", "concerns"])

    for col in ["ingredient_id", "ingredient", "ingredient_url", "functions", "concerns"]:
        if col not in base.columns:
            base[col] = None

    def _first_non_null(series: pd.Series):
        for v in series:
            if v is None:
                continue
            if isinstance(v, float) and pd.isna(v):
                continue
            if isinstance(v, str) and not v.strip():
                continue
            return v
        return None

    merged = (
        base.groupby("ingredient_id", dropna=False)
        .agg(
            ingredient=("ingredient", _first_non_null),
            ingredient_url=("ingredient_url", _first_non_null),
            functions=("functions", _union_str_lists),
            concerns=("concerns", _union_str_lists),
        )
        .reset_index()
    )
    return merged


def _build_ingredients_dim_from_rows(ingredients_df: pd.DataFrame) -> pd.DataFrame:
    """Build a unique ingredient dimension from raw ingredient rows.

    This is used to bootstrap the ingredient dimension cache from an existing
    ingredient_rows.csv when resuming.
    """

    if ingredients_df is None or ingredients_df.empty:
        return pd.DataFrame(columns=["ingredient_id", "ingredient", "ingredient_url", "functions", "concerns"])

    ing = ingredients_df.copy()
    if "ingredient" not in ing.columns:
        raise ValueError("ingredients_df must include an 'ingredient' column")

    ing["ingredient"] = ing["ingredient"].where(pd.notna(ing["ingredient"]), "").astype(str)
    if "ingredient_url" in ing.columns:
        ing["ingredient_url"] = ing["ingredient_url"].where(pd.notna(ing["ingredient_url"]), None)
    else:
        ing["ingredient_url"] = None

    if "ingredient_id" in ing.columns:
        ing["ingredient_id"] = ing["ingredient_id"].where(pd.notna(ing["ingredient_id"]), None)
    else:
        ing["ingredient_id"] = None

    ing.loc[ing["ingredient_id"].isna(), "ingredient_id"] = ing.loc[ing["ingredient_id"].isna()].apply(
        lambda r: make_ingredient_id(r.get("ingredient"), r.get("ingredient_url")), axis=1
    )

    if "functions" not in ing.columns:
        ing["functions"] = []
    if "concerns" not in ing.columns:
        ing["concerns"] = []

    dim = (
        ing.groupby("ingredient_id", dropna=False)
        .agg(
            ingredient=("ingredient", "first"),
            ingredient_url=("ingredient_url", "first"),
            functions=("functions", _union_str_lists),
            concerns=("concerns", _union_str_lists),
        )
        .reset_index()
    )
    return dim


def _read_csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        try:
            return next(r)
        except StopIteration:
            return []


def _migrate_ingredient_rows_cache(path: Path) -> None:
    """Upgrade ingredient_rows.csv cache to include ingredient_id.

    This keeps resume caches forward-compatible without forcing a full re-scrape.
    """

    if not path.exists():
        return
    header = [h.strip() for h in _read_csv_header(path)]
    if not header or "ingredient_id" in header:
        return

    LOGGER.info("Migrating ingredient cache to add ingredient_id: %s", path)
    df = pd.read_csv(path)
    if "ingredient" not in df.columns:
        return

    # Preserve missing values as real nulls so IDs are stable.
    if "ingredient_url" in df.columns:
        df["ingredient_url"] = df["ingredient_url"].where(pd.notna(df["ingredient_url"]), None)
    else:
        df["ingredient_url"] = None
    df["ingredient"] = df["ingredient"].where(pd.notna(df["ingredient"]), "").astype(str)

    df["ingredient_id"] = df.apply(
        lambda r: make_ingredient_id(r.get("ingredient"), r.get("ingredient_url")), axis=1
    )

    # Keep column order stable and append the new column.
    out_cols = list(df.columns)
    if "ingredient_id" in out_cols:
        out_cols.remove("ingredient_id")
    insert_after = "ingredient_url" if "ingredient_url" in out_cols else "ingredient"
    try:
        pos = out_cols.index(insert_after) + 1
    except ValueError:
        pos = len(out_cols)
    out_cols.insert(pos, "ingredient_id")
    df = df[out_cols]

    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _read_csv_set(path: Path, key_col: str, *, where_col: Optional[str] = None, where_value: Optional[str] = None) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if where_col is not None:
                if row.get(where_col) != where_value:
                    continue
            v = (row.get(key_col) or "").strip()
            if v:
                out.add(v)
    return out


def setup_logging(level: int = logging.INFO) -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)


def _default_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


def build_session(
    *,
    total_retries: int = 6,
    backoff_factor: float = 0.8,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """Build a requests session with reasonable retry/backoff defaults."""

    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _sleep_polite(delay: float) -> None:
    if delay <= 0:
        return
    # small jitter to avoid "thundering herd" patterns
    time.sleep(delay + random.uniform(0, min(0.25, delay)))


def _get_soup(
    url: str,
    *,
    session: Optional[requests.Session] = None,
    timeout: tuple[float, float] = (5.0, 30.0),
    max_bytes: int = 10_000_000,
) -> BeautifulSoup:
    sess = session or build_session()
    LOGGER.debug("GET %s", url)
    resp = sess.get(url, headers=_default_headers(), timeout=timeout, stream=True)
    # If we get an error after retries, raise for visibility.
    resp.raise_for_status()

    # Stream content into memory with a hard cap to prevent runaway responses
    # from crashing the whole job.
    buf = bytearray()
    for chunk in resp.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise RuntimeError(f"Response too large (> {max_bytes} bytes) for {url}")

    # Decode safely (avoid MemoryError / bad encodings). For HTML, utf-8 with
    # replacement is good enough for parsing.
    html = bytes(buf).decode("utf-8", errors="replace")
    return BeautifulSoup(html, "lxml")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _dedupe(xs: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        x = _norm(x)
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def _clean_concern(s: str) -> str:
    s = _norm(s)
    if not s:
        return ""
    # Some pages concatenate an explanatory paragraph after the last bullet.
    s = re.split(r"\bEWG has reviewed\b", s, maxsplit=1, flags=re.IGNORECASE)[0]
    s = re.split(r"\bVisit the ingredient page\b", s, maxsplit=1, flags=re.IGNORECASE)[0]
    return _norm(s)


@dataclass(frozen=True)
class ProductRow:
    category: str
    company: Optional[str]
    product: str
    url: str


def scrape_ewg_category(
    *,
    category: str = "Anti-aging",
    page: int = 1,
    timeout: tuple[float, float] = (5.0, 30.0),
    session: Optional[requests.Session] = None,
) -> list[ProductRow]:
    """Scrape EWG SkinDeep category listing for product name + company/brand."""

    category = category.strip()
    if not category:
        raise ValueError("category must be non-empty")
    if page < 1:
        raise ValueError("page must be >= 1")

    if page == 1:
        url = f"{BASE}/skindeep/browse/category/{category}/"
    else:
        url = f"{BASE}/skindeep/browse/category/{category}/?category={category}&page={page}"

    soup = _get_soup(url, session=session, timeout=timeout)
    section = soup.select_one("section.product-listings")
    if section is None:
        raise RuntimeError(
            "Could not find section.product-listings (page layout may have changed or request was blocked)."
        )

    out: list[ProductRow] = []
    for card in section.find_all("div", recursive=False):
        links = [a for a in card.select('a[href*="/skindeep/products/"]') if a.get("href")]
        if not links:
            continue

        link = links[1] if len(links) >= 2 else links[0]
        href = link.get("href")
        full_url = urljoin(BASE, href)

        text_wrapper = link.select_one("div.text-wrapper") or link
        raw_lines = [t.strip() for t in text_wrapper.get_text("\n").splitlines()]
        raw_lines = [t for t in raw_lines if t]

        noise_patterns = [
            r"^EWG VERIFIED\b",
            r"^EWG VERIFIED®\b",
            r"^EWG\s+VERIFIED\b",
            r"^VIEW\b",
        ]
        lines = [
            t
            for t in raw_lines
            if not any(re.search(p, t, flags=re.IGNORECASE) for p in noise_patterns)
        ]

        company = None
        product = None

        company_el = (
            text_wrapper.select_one('[class*="company" i]')
            or text_wrapper.select_one('[class*="brand" i]')
        )
        product_el = (
            text_wrapper.select_one('[class*="product" i]')
            or text_wrapper.select_one('[class*="name" i]')
            or text_wrapper.find(["h3", "h4"])
        )

        if company_el:
            company = company_el.get_text(" ", strip=True)
        if product_el:
            product = product_el.get_text(" ", strip=True)

        if not (company and product):
            if len(lines) >= 2:
                company = company or lines[0]
                product = product or lines[1]
            elif len(lines) == 1:
                product = product or lines[0]

        company = _norm(company) if company else None
        product = _norm(product) if product else None

        if not product:
            continue

        out.append(ProductRow(category=category, company=company, product=product, url=full_url))

    return out


def discover_category_total_pages(
    *,
    category: str = "Anti-aging",
    timeout: tuple[float, float] = (5.0, 30.0),
    session: Optional[requests.Session] = None,
) -> int:
    """Best-effort: infer the last page number from pagination links."""
    category = category.strip()
    url = f"{BASE}/skindeep/browse/category/{category}/"
    soup = _get_soup(url, session=session, timeout=timeout)

    # Common patterns: links containing "page=" with the max page number.
    page_nums: list[int] = []
    for a in soup.select('a[href*="page="]'):
        href = a.get("href") or ""
        m = re.search(r"[?&]page=(\d+)", href)
        if m:
            try:
                page_nums.append(int(m.group(1)))
            except ValueError:
                pass

    return max(page_nums) if page_nums else 1


def scrape_ewg_category_products(
    *,
    category: str = "Anti-aging",
    start_page: int = 1,
    end_page: Optional[int] = None,
    max_pages: Optional[int] = None,
    max_products: Optional[int] = None,
    delay: float = 0.25,
    timeout: tuple[float, float] = (5.0, 30.0),
    session: Optional[requests.Session] = None,
) -> list[ProductRow]:
    """Scrape many category pages into a single list of products."""

    if start_page < 1:
        raise ValueError("start_page must be >= 1")

    sess = session or build_session()

    if end_page is None:
        inferred = discover_category_total_pages(category=category, timeout=timeout, session=sess)
        end_page = inferred

    if max_pages is not None:
        end_page = min(end_page, start_page + max_pages - 1)

    out: list[ProductRow] = []
    seen_urls: set[str] = set()

    for p in range(start_page, end_page + 1):
        LOGGER.info("Scraping category=%s page=%s", category, p)
        try:
            rows = scrape_ewg_category(category=category, page=p, timeout=timeout, session=sess)
        except (requests.RequestException, RuntimeError, MemoryError) as exc:
            LOGGER.warning(
                "Stopping category=%s at page=%s due to error: %s",
                category,
                p,
                exc,
            )
            break

        if not rows:
            LOGGER.info("No products found for category=%s page=%s; stopping pagination", category, p)
            break
        for r in rows:
            if r.url in seen_urls:
                continue
            seen_urls.add(r.url)
            out.append(r)
            if max_products is not None and len(out) >= max_products:
                return out
        _sleep_polite(delay)

    return out


def scrape_ewg_product_ingredients_from_product_page(
    product_url: str,
    *,
    max_ingredients: Optional[int] = None,
    timeout: tuple[float, float] = (5.0, 30.0),
    session: Optional[requests.Session] = None,
) -> list[dict]:
    """Scrape ingredient name + functions + concerns from the product page itself."""

    soup = _get_soup(product_url, session=session, timeout=timeout)

    container = soup.select_one("section.ingredient-scores > section") or soup.select_one(
        "section.ingredient-scores"
    )
    if container is None:
        # Fallback: find 'Ingredient List' heading and take the nearest section/table
        heading = None
        for h in soup.find_all(["h1", "h2", "h3"]):
            if re.search(r"^ingredient\s+list$", h.get_text(" ", strip=True), flags=re.IGNORECASE):
                heading = h
                break
        if heading:
            container = heading.find_parent("section") or heading.find_parent()
    if container is None:
        raise RuntimeError("Could not locate ingredient list section on the product page.")

    table = container.select_one("table.table-ingredient-concerns") or container.select_one("table")
    if table is None:
        raise RuntimeError("Could not find an ingredient table on the product page.")

    def _is_ingredient_header(tr) -> bool:
        txt = _norm(tr.get_text(" ", strip=True))
        if not txt:
            return False
        if re.search(
            r"FUNCTION\(S\)|\bCONCERNS\b|LEARN\s+MORE|EXPAND\s+CONTENT",
            txt,
            flags=re.IGNORECASE,
        ):
            return False
        if len(txt) > 140:
            return False
        return True

    results: list[dict] = []
    current: Optional[dict] = None

    for tr in table.select("tr"):
        if _is_ingredient_header(tr):
            if current is not None:
                current["functions"] = _dedupe(current.get("functions", []))
                current["concerns"] = _dedupe(current.get("concerns", []))
                results.append(current)
                if max_ingredients is not None and len(results) >= max_ingredients:
                    current = None
                    break

            current = {
                "ingredient": _norm(tr.get_text(" ", strip=True)),
                "ingredient_url": None,
                "functions": [],
                "concerns": [],
            }
            a = tr.select_one('a[href*="/skindeep/ingredients/"]')
            if a and a.get("href"):
                current["ingredient_url"] = urljoin(BASE, a["href"])
            continue

        if current is None:
            continue

        if current.get("ingredient_url") is None:
            a = tr.select_one('a[href*="/skindeep/ingredients/"]')
            if a and a.get("href"):
                current["ingredient_url"] = urljoin(BASE, a["href"])

        found_label = False
        for cell in tr.find_all(["td", "th"], recursive=True):
            label = _norm(cell.get_text(" ", strip=True))
            if re.fullmatch(r"FUNCTION\(S\)", label, flags=re.IGNORECASE):
                found_label = True
                val = cell.find_next_sibling(["td", "th"])
                if val:
                    func_txt = _norm(val.get_text(" ", strip=True))
                    current["functions"].extend([x.strip() for x in func_txt.split(",") if x.strip()])
            elif re.fullmatch(r"CONCERNS", label, flags=re.IGNORECASE):
                found_label = True
                val = cell.find_next_sibling(["td", "th"])
                if val:
                    lis = [_norm(li.get_text(" ", strip=True)) for li in val.select("li")]
                    lis = [c for c in (_clean_concern(x) for x in lis) if c]
                    if lis:
                        current["concerns"].extend(lis)
                    else:
                        text = val.get_text(" ", strip=True)
                        if "•" in text or "\u2022" in text:
                            parts = [_norm(p) for p in re.split(r"[•\u2022]", text)]
                            parts = [
                                p
                                for p in parts
                                if p and not re.fullmatch(r"CONCERNS", p, flags=re.IGNORECASE)
                            ]
                            current["concerns"].extend([c for c in (_clean_concern(x) for x in parts) if c])

        if not found_label:
            row_text = _norm(tr.get_text(" ", strip=True))
            if re.search(r"FUNCTION\(S\)", row_text, flags=re.IGNORECASE):
                m = re.search(
                    r"FUNCTION\(S\)\s*(.*?)(?:\bCONCERNS\b|$)", row_text, flags=re.IGNORECASE
                )
                if m:
                    func_txt = _norm(m.group(1))
                    current["functions"].extend([x for x in (t.strip() for t in func_txt.split(",")) if x])

            if re.search(r"\bCONCERNS\b", row_text, flags=re.IGNORECASE) and (
                "•" in row_text or "\u2022" in row_text
            ):
                m = re.search(r"\bCONCERNS\b\s*(.*)$", row_text, flags=re.IGNORECASE)
                if m:
                    parts = [_norm(p) for p in re.split(r"[•\u2022]", m.group(1))]
                    current["concerns"].extend([c for c in (_clean_concern(x) for x in parts) if c])

    if current is not None and (max_ingredients is None or len(results) < max_ingredients):
        current["functions"] = _dedupe(current.get("functions", []))
        current["concerns"] = _dedupe(current.get("concerns", []))
        results.append(current)

    # Final cleanup
    for r in results:
        r["functions"] = _dedupe(r.get("functions", []))
        r["concerns"] = _dedupe(r.get("concerns", []))

    return results


def scrape_categories_to_dataframes(
    *,
    categories: Optional[Iterable[str]] = None,
    start_page: int = 1,
    end_page: Optional[int] = None,
    max_pages: Optional[int] = None,
    max_products_per_category: Optional[int] = None,
    max_ingredients_per_product: Optional[int] = None,
    delay: float = 0.35,
    timeout: tuple[float, float] = (5.0, 30.0),
    state_dir: Optional[str] = None,
    resume: bool = False,
    retry_failed: bool = False,
    refresh_products: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Iterate over categories and return final (products_df, ingredients_df).

    products_df columns: category, company, product, url
    ingredients_df columns: category, company, product, product_url, ingredient, ingredient_url, ingredient_id, functions, concerns
    """

    setup_logging()
    session = build_session()

    state_path = Path(state_dir) if state_dir else None
    if resume and state_path is None:
        raise ValueError("resume=True requires state_dir")

    products_cache_path = state_path / "products.csv" if state_path else None
    ingredient_rows_cache_path = state_path / "ingredient_rows.csv" if state_path else None
    product_status_path = state_path / "product_status.csv" if state_path else None
    ingredients_dim_cache_path = state_path / "ingredients_dim_cache.csv" if state_path else None

    cats = list(categories) if categories is not None else list(DEFAULT_CATEGORIES)
    cats = [c.strip() for c in cats if str(c).strip()]
    if not cats:
        raise ValueError("No categories configured. Set DEFAULT_CATEGORIES or pass categories=...")

    products: list[ProductRow] = []

    # Resume: reuse cached products list so we don't have to re-scrape category pages.
    # If you previously ran with limits (e.g. --max-pages 1) the cache might be partial,
    # so allow forcing a refresh.
    if resume and (not refresh_products) and products_cache_path and products_cache_path.exists():
        LOGGER.info("Resuming: loading cached products list from %s", products_cache_path)
        products_df_cached = pd.read_csv(products_cache_path)
        for _, row in products_df_cached.iterrows():
            products.append(
                ProductRow(
                    category=str(row.get("category", "")),
                    company=None if pd.isna(row.get("company")) else str(row.get("company")),
                    product=str(row.get("product", "")),
                    url=str(row.get("url", "")),
                )
            )
    else:
        for cat in cats:
            products.extend(
                scrape_ewg_category_products(
                    category=cat,
                    start_page=start_page,
                    end_page=end_page,
                    max_pages=max_pages,
                    max_products=max_products_per_category,
                    delay=delay,
                    timeout=timeout,
                    session=session,
                )
            )

        # Cache products for resume
        if state_path and products_cache_path:
            _ensure_dir(state_path)
            pd.DataFrame([p.__dict__ for p in products]).to_csv(products_cache_path, index=False)
            LOGGER.info("Saved products cache to %s (%s rows)", products_cache_path, len(products))

    products_df = pd.DataFrame([p.__dict__ for p in products])

    ingredient_rows: list[dict] = []

    already_ok: set[str] = set()
    already_failed: set[str] = set()
    if resume and product_status_path:
        already_ok = _read_csv_set(product_status_path, "product_url", where_col="status", where_value="ok")
        already_failed = _read_csv_set(product_status_path, "product_url", where_col="status", where_value="error")
        LOGGER.info(
            "Resuming: %s products already ok, %s products already error",
            len(already_ok),
            len(already_failed),
        )

    # Load cached ingredient rows so we can keep building from partial state.
    if resume and ingredient_rows_cache_path and ingredient_rows_cache_path.exists():
        _migrate_ingredient_rows_cache(ingredient_rows_cache_path)
        LOGGER.info("Resuming: loading cached ingredient rows from %s", ingredient_rows_cache_path)
        ingredient_rows = pd.read_csv(ingredient_rows_cache_path).to_dict(orient="records")

        # If we have ingredient rows but no ingredient dimension cache yet, bootstrap it.
        # This enables the "only enrich new ingredients" behavior even on old state dirs.
        if ingredients_dim_cache_path and not ingredients_dim_cache_path.exists():
            LOGGER.info(
                "Bootstrapping ingredient dimension cache from %s -> %s",
                ingredient_rows_cache_path,
                ingredients_dim_cache_path,
            )
            dim_boot = _build_ingredients_dim_from_rows(pd.DataFrame(ingredient_rows))
            _ensure_dir(ingredients_dim_cache_path.parent)
            dim_boot_out = dim_boot.copy()
            dim_boot_out["functions"] = dim_boot_out["functions"].apply(json.dumps)
            dim_boot_out["concerns"] = dim_boot_out["concerns"].apply(json.dumps)
            dim_boot_out.to_csv(ingredients_dim_cache_path, index=False)

    # Ingredient dimension cache: used to avoid re-enriching already-known ingredient_ids.
    known_ingredient_ids: set[str] = set()
    if resume and ingredients_dim_cache_path and ingredients_dim_cache_path.exists():
        dim_cached = _load_ingredients_dim_cache(ingredients_dim_cache_path)
        known_ingredient_ids = {
            str(v)
            for v in dim_cached.get("ingredient_id", pd.Series(dtype=object)).dropna().tolist()
            if str(v).strip()
        }

    ingredient_cache_fields = [
        "category",
        "company",
        "product",
        "product_url",
        "ingredient",
        "ingredient_url",
        "ingredient_id",
        "functions",
        "concerns",
    ]
    ingredients_dim_cache_fields = [
        "ingredient_id",
        "ingredient",
        "ingredient_url",
        "functions",
        "concerns",
    ]
    status_fields = ["product_url", "status", "error"]

    for idx, p in enumerate(products, start=1):
        if resume and p.url in already_ok:
            continue
        if resume and (not retry_failed) and p.url in already_failed:
            continue

        LOGGER.info("Scraping ingredients %s/%s | %s", idx, len(products), p.url)
        try:
            ing = scrape_ewg_product_ingredients_from_product_page(
                p.url,
                max_ingredients=max_ingredients_per_product,
                timeout=timeout,
                session=session,
            )
        except (requests.RequestException, RuntimeError, MemoryError) as exc:
            LOGGER.warning("Failed product ingredients for %s: %s", p.url, exc)
            if product_status_path:
                _append_csv_row(
                    product_status_path,
                    status_fields,
                    {"product_url": p.url, "status": "error", "error": str(exc)},
                )
            _sleep_polite(delay)
            continue

        if product_status_path:
            _append_csv_row(
                product_status_path,
                status_fields,
                {"product_url": p.url, "status": "ok", "error": ""},
            )

        for row in ing:
            ingredient_id = make_ingredient_id(row.get("ingredient"), row.get("ingredient_url"))
            is_new_ingredient = ingredient_id not in known_ingredient_ids
            record = {
                "category": p.category,
                "company": p.company,
                "product": p.product,
                "product_url": p.url,
                "ingredient_id": ingredient_id,
                # Always keep ingredient + URL so the junction can be rebuilt.
                "ingredient": row.get("ingredient"),
                "ingredient_url": row.get("ingredient_url"),
                # Only enrich functions/concerns the first time we see this ingredient_id
                # (across products and across resume runs when the dim cache exists).
                "functions": row.get("functions") if is_new_ingredient else [],
                "concerns": row.get("concerns") if is_new_ingredient else [],
            }
            ingredient_rows.append(record)

            if is_new_ingredient and ingredients_dim_cache_path:
                known_ingredient_ids.add(ingredient_id)
                cache_dim_row = {
                    "ingredient_id": ingredient_id,
                    "ingredient": row.get("ingredient"),
                    "ingredient_url": row.get("ingredient_url"),
                    "functions": json.dumps(_as_list(row.get("functions"))),
                    "concerns": json.dumps(_as_list(row.get("concerns"))),
                }
                _append_csv_row(ingredients_dim_cache_path, ingredients_dim_cache_fields, cache_dim_row)

            if ingredient_rows_cache_path:
                # Persist incrementally so we can resume without repeating work.
                # Store functions/concerns as JSON strings for easy reload.
                cache_row = dict(record)
                if isinstance(cache_row.get("functions"), list):
                    cache_row["functions"] = json.dumps(cache_row["functions"])
                if isinstance(cache_row.get("concerns"), list):
                    cache_row["concerns"] = json.dumps(cache_row["concerns"])
                _append_csv_row(ingredient_rows_cache_path, ingredient_cache_fields, cache_row)

        _sleep_polite(delay)

    ingredients_df = pd.DataFrame(ingredient_rows)
    return products_df, ingredients_df


def make_ingredient_id(ingredient: str, ingredient_url: Optional[str] = None) -> str:
    """Deterministic ID for an ingredient.

    Uses a stable hash of normalized name + URL (if present) so IDs remain
    consistent across runs.
    """

    key = _norm(str(ingredient)).lower()
    if ingredient_url:
        key = f"{key}|{str(ingredient_url).strip()}"
    digest = hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()  # noqa: S324
    return f"ing_{digest[:16]}"


def build_sql_like_tables(
    *,
    products_df: pd.DataFrame,
    ingredients_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create SQL-friendly tables.

    Returns:
      - products_out: products with a JSON string column `ingredient_ids`
      - ingredients_dim: one row per ingredient with `ingredient_id`
      - product_ingredients: junction table (product_url, ingredient_id)
    """

    if ingredients_df.empty:
        products_out = products_df.copy()
        products_out["ingredient_ids"] = "[]"
        ingredients_dim = pd.DataFrame(
            columns=["ingredient_id", "ingredient", "ingredient_url", "functions", "concerns"]
        )
        product_ingredients = pd.DataFrame(columns=["product_url", "ingredient_id"])
        return products_out, ingredients_dim, product_ingredients

    ing = ingredients_df.copy()
    if "ingredient" not in ing.columns:
        raise ValueError("ingredients_df must include an 'ingredient' column")

    # Keep missing values as real nulls (not the literal string 'nan'), otherwise
    # make_ingredient_id() will generate unstable IDs and fragment deduplication.
    ing["ingredient"] = ing["ingredient"].where(pd.notna(ing["ingredient"]), "").astype(str)
    if "ingredient_url" in ing.columns:
        ing["ingredient_url"] = ing["ingredient_url"].where(pd.notna(ing["ingredient_url"]), None)
    else:
        ing["ingredient_url"] = None

    if "ingredient_id" in ing.columns:
        ing["ingredient_id"] = ing["ingredient_id"].where(pd.notna(ing["ingredient_id"]), None)

    # Prefer an existing ingredient_id (e.g., from state cache), otherwise compute.
    missing_ids = ("ingredient_id" not in ing.columns) or ing["ingredient_id"].isna().all()
    if missing_ids:
        ing["ingredient_id"] = ing.apply(
            lambda r: make_ingredient_id(r.get("ingredient"), r.get("ingredient_url")), axis=1
        )
    else:
        ing.loc[ing["ingredient_id"].isna(), "ingredient_id"] = ing.loc[
            ing["ingredient_id"].isna()
        ].apply(lambda r: make_ingredient_id(r.get("ingredient"), r.get("ingredient_url")), axis=1)

    # Junction table (proper relational model)
    product_ingredients = (
        ing[["product_url", "ingredient_id"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Ingredient dimension: aggregate functions/concerns across occurrences
    def _union_lists(series: pd.Series) -> list[str]:
        return _union_str_lists(series.tolist())

    ingredients_dim = (
        ing.groupby("ingredient_id", dropna=False)
        .agg(
            ingredient=("ingredient", "first"),
            ingredient_url=("ingredient_url", "first"),
            functions=("functions", _union_lists) if "functions" in ing.columns else ("ingredient", lambda _: []),
            concerns=("concerns", _union_lists) if "concerns" in ing.columns else ("ingredient", lambda _: []),
        )
        .reset_index()
    )

    # Products with ingredient_ids as JSON (convenient, but junction table is better)
    products_out = products_df.copy()
    product_url_col = "product_url" if "product_url" in products_out.columns else "url"
    if product_url_col not in products_out.columns:
        raise ValueError("products_df must have either 'url' or 'product_url' column")

    ids_by_product = (
        product_ingredients.groupby("product_url")["ingredient_id"]
        .apply(lambda s: json.dumps(sorted(set(s.tolist()))))
        .to_dict()
    )
    products_out["ingredient_ids"] = products_out[product_url_col].map(ids_by_product).fillna("[]")

    return products_out, ingredients_dim, product_ingredients


def _parse_categories_arg(raw: str) -> list[str]:
    # Allow comma-separated or repeated usage in a shell.
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def main(argv: Optional[list[str]] = None) -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description="EWG SkinDeep one-shot scraper")
    parser.add_argument(
        "--categories",
        required=False,
        default=None,
        help="Optional override. Comma-separated categories (as used in the URL). Defaults to DEFAULT_CATEGORIES in this file.",
    )
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-products-per-category", type=int, default=None)
    parser.add_argument("--max-ingredients-per-product", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous run using cached state in --state-dir (skips already processed products).",
    )
    parser.add_argument(
        "--refresh-products",
        action="store_true",
        help="When resuming, re-scrape category pages to rebuild the products list (overwrites cached products.csv).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="When resuming, retry products previously marked as error.",
    )
    parser.add_argument(
        "--only-build-from-state",
        action="store_true",
        help="Skip scraping; load cached products/ingredient rows from --state-dir and only build/write outputs.",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default=".ewg_state",
        help="Directory for resume cache/state (products.csv, ingredient_rows.csv, product_status.csv).",
    )
    parser.add_argument(
        "--products-with-ingredient-ids-csv",
        type=str,
        default="products_with_ingredient_ids.csv",
        help="Writes products + a JSON list column `ingredient_ids`.",
    )
    parser.add_argument(
        "--ingredients-dim-csv",
        type=str,
        default="ingredients_dim.csv",
        help="Writes unique ingredients dimension with `ingredient_id` (deduped).",
    )
    parser.add_argument(
        "--write-raw",
        action="store_true",
        help="Optional. Also write the raw scrape outputs (products/ingredients rows).",
    )
    parser.add_argument(
        "--products-csv",
        type=str,
        default=None,
        help="(Raw) Optional. Writes raw products rows. Only used with --write-raw.",
    )
    parser.add_argument(
        "--ingredients-csv",
        type=str,
        default=None,
        help="(Raw) Optional. Writes raw ingredient rows. Only used with --write-raw.",
    )
    parser.add_argument(
        "--product-ingredients-csv",
        type=str,
        default=None,
        help="Optional. Writes junction table (product_url, ingredient_id).",
    )
    args = parser.parse_args(argv)

    cats = _parse_categories_arg(args.categories) if args.categories else None

    if args.only_build_from_state:
        state_path = Path(args.state_dir)
        products_cache_path = state_path / "products.csv"
        ingredient_rows_cache_path = state_path / "ingredient_rows.csv"
        ingredients_dim_cache_path = state_path / "ingredients_dim_cache.csv"
        if not products_cache_path.exists():
            raise FileNotFoundError(f"Missing {products_cache_path}. Run a scrape first (or disable --only-build-from-state).")
        if not ingredient_rows_cache_path.exists():
            raise FileNotFoundError(f"Missing {ingredient_rows_cache_path}. Run a scrape first (or disable --only-build-from-state).")

        _migrate_ingredient_rows_cache(ingredient_rows_cache_path)
        products_df = pd.read_csv(products_cache_path)
        ingredients_df = pd.read_csv(ingredient_rows_cache_path)
        LOGGER.info(
            "Loaded cached state from %s (products=%s, ingredient_rows=%s)",
            state_path,
            len(products_df),
            len(ingredients_df),
        )
        ingredients_dim_cache_df = (
            _load_ingredients_dim_cache(ingredients_dim_cache_path)
            if ingredients_dim_cache_path.exists()
            else pd.DataFrame(columns=["ingredient_id", "ingredient", "ingredient_url", "functions", "concerns"])
        )
    else:
        products_df, ingredients_df = scrape_categories_to_dataframes(
            categories=cats,
            max_pages=args.max_pages,
            max_products_per_category=args.max_products_per_category,
            max_ingredients_per_product=args.max_ingredients_per_product,
            delay=args.delay,
            state_dir=args.state_dir,
            resume=args.resume,
            retry_failed=args.retry_failed,
            refresh_products=args.refresh_products,
        )
        ingredients_dim_cache_df = (
            _load_ingredients_dim_cache(Path(args.state_dir) / "ingredients_dim_cache.csv")
            if args.state_dir
            else pd.DataFrame(columns=["ingredient_id", "ingredient", "ingredient_url", "functions", "concerns"])
        )

    products_out, ingredients_dim, product_ingredients = build_sql_like_tables(
        products_df=products_df,
        ingredients_df=ingredients_df,
    )

    # Merge any cached ingredient dimension (from prior runs) so the output stays complete
    # even if we skipped re-enriching duplicates during this run.
    ingredients_dim = _merge_ingredients_dim(ingredients_dim_cache_df, ingredients_dim)

    # User-facing alias column (requested name; keep existing ingredient_ids for compatibility).
    products_out["ingrediants"] = products_out.get("ingredient_ids", "[]")

    # Final expected outputs
    products_out.to_csv(args.products_with_ingredient_ids_csv, index=False)
    LOGGER.info(
        "Wrote %s (%s rows)", args.products_with_ingredient_ids_csv, len(products_out)
    )
    ingredients_dim.to_csv(args.ingredients_dim_csv, index=False)
    LOGGER.info("Wrote %s (%s rows)", args.ingredients_dim_csv, len(ingredients_dim))

    # Optional additional outputs
    if args.product_ingredients_csv:
        product_ingredients.to_csv(args.product_ingredients_csv, index=False)
        LOGGER.info("Wrote %s (%s rows)", args.product_ingredients_csv, len(product_ingredients))

    if args.write_raw:
        if args.products_csv:
            products_df.to_csv(args.products_csv, index=False)
            LOGGER.info("Wrote %s (%s rows)", args.products_csv, len(products_df))
        if args.ingredients_csv:
            ingredients_df.to_csv(args.ingredients_csv, index=False)
            LOGGER.info("Wrote %s (%s rows)", args.ingredients_csv, len(ingredients_df))

    LOGGER.info("Done. products=%s ingredients=%s", len(products_df), len(ingredients_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
