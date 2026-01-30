"""
Enrichment Script: Transform ingredients_dim_cache.csv to SQL seed data
Uses DeepSeek API to add missing fields: skin_type_compatibility, interactions, recommendation_time, effects
"""

import csv
import json
import os
import re
import time
from pathlib import Path
from openai import OpenAI

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CACHE_FILE = PROJECT_ROOT / "ingredients_dim_cache.csv"
OUTPUT_SQL = PROJECT_ROOT / "webapp" / "sql" / "seed_ingredients.sql"

# Load from .env.txt if exists
def load_env():
    env_file = PROJECT_ROOT / ".env.txt"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
                    if key == "DEEPSEEK_API_KEY":
                        print(f"   ✓ Loaded DEEPSEEK_API_KEY: {value[:10]}...")

load_env()
DEEPSEEK_API_KEY = "sk-f86dab0c76f44d09a012e691dc6243d6"

# DeepSeek client
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


def clean_ingredient_name(raw_name: str) -> tuple[str, str]:
    """
    Extract clean ingredient name from EWG format
    Input: "ALOE BARBADENSIS (ALOE VERA) LEAF JUICE Data Availability: Good"
    Output: ("aloe_vera", "Aloe Vera Leaf Juice")
    """
    # Remove "Data Availability: ..." suffix
    name = re.sub(r'\s*Data Availability:.*$', '', raw_name, flags=re.IGNORECASE)
    
    # Extract common name from parentheses if exists
    match = re.search(r'\(([^)]+)\)', name)
    if match:
        common_name = match.group(1)
    else:
        common_name = name
    
    # Create display name (title case)
    display_name = name.title()
    
    # Create snake_case ID
    # Keep only alphanumeric and spaces, then convert
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', common_name)
    ingredient_id = clean.lower().strip().replace(' ', '_')
    
    # Limit length
    if len(ingredient_id) > 30:
        ingredient_id = ingredient_id[:30]
    
    return ingredient_id, display_name


# Mapping EWG functions to our standardized effects
FUNCTION_TO_EFFECT = {
    # Hydration
    "humectant": "hydration",
    "moisturising": "hydration",
    "moisturizing": "hydration",
    "skin-conditioning agent - humectant": "hydration",
    # Anti-aging
    "antioxidant": "antioxidant",
    "anti-oxidant": "antioxidant",
    "uv absorber": "antioxidant",
    "ultraviolet light absorber": "antioxidant",
    # Soothing
    "skin conditioning": "soothing",
    "skin-conditioning agent - miscellaneous": "soothing",
    "skin protecting": "barrier_repair",
    "skin protectant": "barrier_repair",
    # Exfoliation
    "keratolytic": "exfoliation",
    "exfoliant": "exfoliation",
    # Oil control / Astringent
    "astringent": "oil_control",
    "drug astringent - skin protectant drugs": "oil_control",
    # Plumping / Emollient
    "emollient": "plumping",
    "skin-conditioning agent - emollient": "plumping",
    "skin-conditioning agent - occlusive": "barrier_repair",
    # Brightening
    "tonic": "brightening",
    "refreshing": "brightening",
    # Anti-acne
    "antimicrobial": "anti_acne",
    "antidandruff": "anti_acne",
}


def map_functions_to_effects(functions_str: str) -> list[str]:
    """Convert EWG functions to our standardized effects"""
    try:
        # Parse the JSON-like string
        functions = json.loads(functions_str.replace("'", '"')) if functions_str else []
    except:
        functions = []
    
    effects = set()
    for func in functions:
        func_lower = func.lower().strip()
        if func_lower in FUNCTION_TO_EFFECT:
            effects.add(FUNCTION_TO_EFFECT[func_lower])
    
    # Default to hydration if nothing mapped
    if not effects:
        effects.add("hydration")
    
    return list(effects)


def load_cache(filepath: Path) -> list[dict]:
    """Load ingredients from CSV cache"""
    ingredients = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ingredients.append(row)
    return ingredients


def build_enrichment_prompt(ingredients_batch: list[dict]) -> str:
    """Build prompt for DeepSeek enrichment - only for fields we don't have"""
    
    ingredient_list = []
    for ing in ingredients_batch:
        name = ing['ingredient']
        concerns = ing.get('concerns', '[]')
        ingredient_list.append(f"- {name}\n  Concerns: {concerns}")
    
    prompt = f"""You are a cosmetic chemist expert. For each ingredient below, provide:

1. skin_type_compatibility: Array of skin types this ingredient is suitable for
   Options: "all", "oily", "dry", "combination", "sensitive", "normal", "acne_prone"
   Consider the concerns listed to determine suitability.

2. interactions: Array of ingredient names that have NEGATIVE interactions (conflicts) with this ingredient
   Use simple lowercase names like: "retinol", "vitamin_c", "niacinamide", "glycolic_acid", "salicylic_acid", "benzoyl_peroxide", "aha", "bha"
   Only list ingredients that should NOT be used together. If none, return empty array.

3. recommendation_time: When to use this ingredient
   Options: "morning", "evening", or both ["morning", "evening"]
   Consider: photosensitizing ingredients = evening only, antioxidants = morning preferred

INGREDIENTS TO ANALYZE:
{chr(10).join(ingredient_list)}

Return ONLY a valid JSON array with this structure (no markdown, no explanation):
[
  {{
    "original_name": "EXACT NAME FROM INPUT",
    "skin_type_compatibility": ["all"],
    "interactions": [],
    "recommendation_time": ["morning", "evening"]
  }}
]
"""
    return prompt


def enrich_batch(ingredients_batch: list[dict]) -> list[dict]:
    """Call DeepSeek to enrich a batch of ingredients"""
    
    prompt = build_enrichment_prompt(ingredients_batch)
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a cosmetic chemistry expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean markdown if present
        if content.startswith("```"):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        
        enriched = json.loads(content)
        return enriched
        
    except Exception as e:
        print(f"  ❌ Error enriching batch: {e}")
        return []


def generate_sql(ingredients: list[dict], enriched_map: dict) -> str:
    """Generate SQL INSERT statements"""
    
    sql_lines = [
        "-- ============================================================================",
        "-- SEED DATA: Enriched ingredients from EWG cache",
        "-- Generated by enrich_cache_to_sql.py",
        "-- ============================================================================",
        "",
        "-- Clear existing test data",
        "TRUNCATE TABLE active_ingredients_database;",
        "",
        "INSERT INTO active_ingredients_database ",
        "(ingredient_id, ingredient_name, display_name, product_type, skin_type_compatibility, interactions, recommendation_time, effects)",
        "VALUES"
    ]
    
    values = []
    seen_ids = set()
    
    for ing in ingredients:
        raw_name = ing['ingredient']
        ingredient_id, display_name = clean_ingredient_name(raw_name)
        
        # Skip duplicates
        if ingredient_id in seen_ids or not ingredient_id:
            continue
        seen_ids.add(ingredient_id)
        
        # Get enriched data from DeepSeek
        enriched = enriched_map.get(raw_name, {})
        
        skin_types = enriched.get('skin_type_compatibility', ['all'])
        interactions = enriched.get('interactions', [])
        rec_time = enriched.get('recommendation_time', ['morning', 'evening'])
        
        # Map effects from existing functions column (no DeepSeek needed!)
        effects = map_functions_to_effects(ing.get('functions', '[]'))
        
        # Format arrays for PostgreSQL
        def pg_array(arr):
            if not arr:
                return "ARRAY[]::TEXT[]"
            escaped = [s.replace("'", "''") for s in arr]
            quoted = ["'" + s + "'" for s in escaped]
            return "ARRAY[" + ", ".join(quoted) + "]"
        
        # Escape strings
        safe_id = ingredient_id.replace("'", "''")
        safe_display = display_name.replace("'", "''")[:100]
        
        value = "('" + safe_id + "', '" + safe_id + "', '" + safe_display + "', 'serum', " + pg_array(skin_types) + ", " + pg_array(interactions) + ", " + pg_array(rec_time) + ", " + pg_array(effects) + ")"
        values.append(value)
    
    sql_lines.append(",\n".join(values) + ";")
    sql_lines.append("")
    sql_lines.append(f"-- Total: {len(values)} ingredients loaded")
    sql_lines.append("DO $$ BEGIN RAISE NOTICE '✅ Loaded {} enriched ingredients'; END $$;".format(len(values)))
    
    return "\n".join(sql_lines)


def main():
    print("=" * 60)
    print("🧪 Ingredient Enrichment Script")
    print("=" * 60)
    
    # Check API key
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("sk-xxx"):
        print("❌ DEEPSEEK_API_KEY not set!")
        print("   Set it with: $env:DEEPSEEK_API_KEY='your-key'")
        return
    
    print(f"   ✓ API Key loaded: {DEEPSEEK_API_KEY[:12]}...")
    
    # Load cache
    print(f"\n📂 Loading {CACHE_FILE}...")
    ingredients = load_cache(CACHE_FILE)
    print(f"   Found {len(ingredients)} ingredients")
    
    # Enrich in batches of 10
    BATCH_SIZE = 10
    enriched_map = {}
    
    print(f"\n🔬 Enriching with DeepSeek (batch size: {BATCH_SIZE})...")
    
    for i in range(0, len(ingredients), BATCH_SIZE):
        batch = ingredients[i:i+BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(ingredients) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"   Batch {batch_num}/{total_batches}...", end=" ", flush=True)
        
        enriched = enrich_batch(batch)
        
        # Map results back by original name
        for item in enriched:
            orig_name = item.get('original_name', '')
            enriched_map[orig_name] = item
        
        print(f"✓ ({len(enriched)} enriched)")
        
        # Rate limiting
        if i + BATCH_SIZE < len(ingredients):
            time.sleep(1)
    
    # Generate SQL
    print(f"\n📝 Generating SQL...")
    sql = generate_sql(ingredients, enriched_map)
    
    # Write output
    with open(OUTPUT_SQL, 'w', encoding='utf-8') as f:
        f.write(sql)
    
    print(f"   ✅ Written to {OUTPUT_SQL}")
    print(f"\n🎉 Done! {len(enriched_map)} ingredients enriched.")
    print("\nNext steps:")
    print("  1. Review the generated SQL")
    print("  2. Run: docker compose exec postgres psql -U airflow -d airflow -f /docker-entrypoint-initdb.d/seed_ingredients.sql")
    print("  3. Or restart containers to auto-load")


if __name__ == "__main__":
    main()
