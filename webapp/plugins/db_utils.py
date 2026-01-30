# plugins/db_utils.py
"""
Utilitaires pour interagir avec PostgreSQL
Gère les requêtes vers la table active_ingredients_database et products_dim
"""
import psycopg2
import re
from typing import List, Dict, Any, Optional, Tuple


def normalize_ingredient_name(raw_name: str) -> str:
    """
    Normalize OCR-detected ingredient name to match database format.
    
    Input: "NIACINAMIDE" or "Aloe Barbadensis (Aloe Vera) Leaf Juice"
    Output: "niacinamide" or "aloe_vera"
    """
    # Remove "Data Availability: ..." suffix if present
    name = re.sub(r'\s*Data Availability:.*$', '', raw_name, flags=re.IGNORECASE)
    
    # Extract common name from parentheses if exists (e.g., "ALOE VERA" from "(ALOE VERA)")
    match = re.search(r'\(([^)]+)\)', name)
    if match:
        common_name = match.group(1)
    else:
        common_name = name
    
    # Create snake_case ID - same logic as enrichment script
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', common_name)
    normalized = clean.lower().strip().replace(' ', '_')
    
    # Limit length to 30 chars to match the enrichment script
    if len(normalized) > 30:
        normalized = normalized[:30]
    
    return normalized


def get_db_connection():
    """
    Crée une connexion à PostgreSQL
    
    Returns:
        Connection object psycopg2
    """
    conn = psycopg2.connect(
        host="postgres",       # Nom du container Docker
        port=5432,             # Port PostgreSQL
        database="airflow",    # Nom de la base
        user="airflow",        # Username
        password="airflow"     # Password
    )
    return conn


def get_ingredients_data(ingredient_names: List[str]) -> List[Dict[str, Any]]:
    """
    Récupère les données d'actifs depuis la base de données
    
    Args:
        ingredient_names: Liste des noms d'actifs 
                         Ex: ['niacinamide', 'retinol']
        
    Returns:
        Liste de dictionnaires contenant les données de chaque actif
        Ex: [
            {
                'ingredient_name': 'niacinamide',
                'display_name': 'Niacinamide',
                'interactions': ['vitamin_c'],
                'effects': ['anti_acne', 'oil_control'],
                ...
            }
        ]
    """
    # Connexion à la base
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Requête SQL pour récupérer les actifs
    query = """
        SELECT 
            ingredient_name,
            display_name,
            product_type,
            skin_type_compatibility,
            interactions,
            recommendation_time,
            effects
        FROM active_ingredients_database
        WHERE ingredient_name = ANY(%s)
    """
    
    # Exécution de la requête
    cursor.execute(query, (ingredient_names,))
    rows = cursor.fetchall()
    
    # Conversion des résultats en liste de dictionnaires
    results = []
    for row in rows:
        results.append({
            'ingredient_name': row[0],
            'display_name': row[1],
            'product_type': row[2],
            'skin_type_compatibility': row[3],
            'interactions': row[4],
            'recommendation_time': row[5],
            'effects': row[6]
        })
    
    # Fermeture de la connexion
    cursor.close()
    conn.close()
    
    return results


def get_ingredients_by_effect(effect: str) -> List[Dict[str, Any]]:
    """
    Trouve tous les actifs qui ont un effet donné
    
    Args:
        effect: Effet recherché 
               Ex: 'anti_acne', 'hydration', 'brightening'
        
    Returns:
        Liste d'actifs qui ont cet effet
        Ex: [
            {
                'ingredient_name': 'niacinamide',
                'display_name': 'Niacinamide',
                'product_type': 'serum',
                'effects': ['anti_acne', 'oil_control']
            }
        ]
    """
    # Connexion à la base
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Requête SQL : cherche si l'effet est dans le tableau effects
    query = """
        SELECT 
            ingredient_name,
            display_name,
            product_type,
            effects
        FROM active_ingredients_database
        WHERE %s = ANY(effects)
    """
    
    # Exécution
    cursor.execute(query, (effect,))
    rows = cursor.fetchall()
    
    # Conversion en dictionnaires
    results = []
    for row in rows:
        results.append({
            'ingredient_name': row[0],
            'display_name': row[1],
            'product_type': row[2],
            'effects': row[3]
        })
    
    # Fermeture
    cursor.close()
    conn.close()
    
    return results


def find_matching_products(scanned_ingredients: List[str], top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Find products from products_dim that best match scanned ingredients.
    
    Args:
        scanned_ingredients: List of ingredient names from OCR (e.g., ["NIACINAMIDE", "GLYCERIN"])
        top_n: Number of top matching products to return
        
    Returns:
        List of matching products with similarity scores
    """
    # Normalize scanned ingredients to database format
    normalized = [normalize_ingredient_name(ing) for ing in scanned_ingredients]
    normalized = [n for n in normalized if n]  # Filter empty
    
    if not normalized:
        return []
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Query to find products with overlapping ingredients and calculate match score
    query = """
        WITH scanned AS (
            SELECT unnest(%s::varchar[]) AS ing_id
        ),
        product_matches AS (
            SELECT 
                p.product_id,
                p.product_name,
                p.product_url,
                p.category,
                p.company,
                p.ingredient_count,
                COUNT(DISTINCT s.ing_id) AS matched_count,
                ARRAY_AGG(DISTINCT s.ing_id) AS matched_ingredients
            FROM products_dim p
            JOIN LATERAL unnest(p.ingredient_ids) AS prod_ing ON TRUE
            JOIN scanned s ON s.ing_id = prod_ing
            GROUP BY p.product_id, p.product_name, p.product_url, p.category, p.company, p.ingredient_count
        )
        SELECT 
            product_id,
            product_name,
            product_url,
            category,
            company,
            ingredient_count,
            matched_count,
            matched_ingredients,
            (matched_count::float / %s) AS match_pct
        FROM product_matches
        ORDER BY match_pct DESC, matched_count DESC
        LIMIT %s
    """
    
    try:
        cursor.execute(query, (normalized, len(normalized), top_n))
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                'product_id': row[0],
                'product_name': row[1],
                'product_url': row[2],
                'category': row[3],
                'company': row[4],
                'ingredient_count': row[5],
                'matched_count': row[6],
                'matched_ingredients': row[7],
                'match_pct': round(row[8] * 100, 1)
            })
        
        return results
        
    except Exception as e:
        print(f"Error finding matching products: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def get_normalized_ingredients_data(raw_ingredient_names: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Normalize ingredient names from OCR and query database.
    Returns both the ingredient data and the normalized names that matched.
    
    Args:
        raw_ingredient_names: List of raw ingredient names from OCR
        
    Returns:
        Tuple of (ingredient_data_list, matched_normalized_names)
    """
    # Normalize each ingredient name
    normalized_names = [normalize_ingredient_name(ing) for ing in raw_ingredient_names]
    normalized_names = [n for n in normalized_names if n]  # Filter empty
    
    if not normalized_names:
        return [], []
    
    # Query database with normalized names
    ingredients_data = get_ingredients_data(normalized_names)
    
    # Get list of matched names
    matched_names = [ing['ingredient_name'] for ing in ingredients_data]
    
    return ingredients_data, matched_names