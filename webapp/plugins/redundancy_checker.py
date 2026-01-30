# plugins/redundancy_checker.py
"""
Détecte les redondances (actifs en double ET effets en double)
"""
from typing import List, Dict, Any


def check_redundancy(enriched_products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Détecte 2 types de redondances :
    1. Même actif dans plusieurs produits
    2. Même effet obtenu par différents actifs
    
    Args:
        enriched_products: Produits avec données enrichies
    
    Returns:
        {
            'hasRedundancy': True/False,
            'count': nombre total,
            'byIngredient': [...],  # Redondances d'actifs
            'byEffect': [...]       # Redondances d'effets
        }
    """
    # ========================================
    # TYPE 1 : Redondance par ACTIF
    # ========================================
    ingredient_redundancy = _check_ingredient_redundancy(enriched_products)
    
    # ========================================
    # TYPE 2 : Redondance par EFFET
    # ========================================
    effect_redundancy = _check_effect_redundancy(enriched_products)
    
    # Combine les 2 types
    all_redundancies = ingredient_redundancy + effect_redundancy
    
    return {
        'hasRedundancy': len(all_redundancies) > 0,
        'count': len(all_redundancies),
        'byIngredient': ingredient_redundancy,
        'byEffect': effect_redundancy,
        'details': all_redundancies  # Pour compatibilité avec le reste du code
    }


def _check_ingredient_redundancy(enriched_products: List[Dict]) -> List[Dict]:
    """
    TYPE 1 : Détecte les actifs présents dans plusieurs produits
    
    Exemple :
    - Niacinamide dans 2 produits → REDONDANCE
    
    Returns:
        Liste de redondances d'actifs
    """
    ingredient_count = {}
    ingredient_products = {}
    
    # Compte combien de fois chaque actif apparaît
    for product in enriched_products:
        product_id = product['userProductId']
        product_name = product.get('productName', product_id)
        
        for ingredient in product.get('activeIngredients', []):
            
            # Récupère le nom d'affichage
            display_name = ingredient
            if ingredient in product.get('enrichedData', {}):
                display_name = product['enrichedData'][ingredient].get('display_name', ingredient)
            
            # Incrémente le compteur
            if ingredient not in ingredient_count:
                ingredient_count[ingredient] = 0
                ingredient_products[ingredient] = {
                    'display_name': display_name,
                    'found_in': []
                }
            
            ingredient_count[ingredient] += 1
            ingredient_products[ingredient]['found_in'].append({
                'productId': product_id,
                'productName': product_name
            })
    
    # Trouve les redondances (actif présent 2+ fois)
    redundancies = []
    for ingredient, count in ingredient_count.items():
        if count > 1:
            info = ingredient_products[ingredient]
            redundancies.append({
                'type': 'ingredient',  # Type de redondance
                'ingredient': ingredient,
                'displayName': info['display_name'],
                'count': count,
                'foundInProducts': info['found_in'],
                'impact': 'low',
                'recommendation': f"{info['display_name']} est présent dans {count} produits. Un seul suffit généralement."
            })
    
    return redundancies


def _check_effect_redundancy(enriched_products: List[Dict]) -> List[Dict]:
    """
    TYPE 2 : Détecte les effets obtenus par plusieurs actifs différents
    
    Exemple :
    - Niacinamide (anti_acne) + Salicylic Acid (anti_acne) → REDONDANCE d'effet
    
    Returns:
        Liste de redondances d'effets
    """
    effect_ingredients = {}  # {effect: [list of ingredients that have it]}
    
    # Collecte tous les effets et les actifs qui les produisent
    for product in enriched_products:
        product_id = product['userProductId']
        product_name = product.get('productName', product_id)
        
        for ingredient in product.get('activeIngredients', []):
            
            # Récupère les effets de cet actif
            if ingredient not in product.get('enrichedData', {}):
                continue
            
            ing_data = product['enrichedData'][ingredient]
            effects = ing_data.get('effects', [])
            display_name = ing_data.get('display_name', ingredient)
            
            # Pour chaque effet
            for effect in effects:
                if effect not in effect_ingredients:
                    effect_ingredients[effect] = []
                
                # Ajoute cet actif à la liste (évite doublons)
                if ingredient not in [item['ingredient'] for item in effect_ingredients[effect]]:
                    effect_ingredients[effect].append({
                        'ingredient': ingredient,
                        'displayName': display_name,
                        'productId': product_id,
                        'productName': product_name
                    })
    
    # Trouve les redondances (effet obtenu par 2+ actifs)
    redundancies = []
    for effect, ingredients_list in effect_ingredients.items():
        if len(ingredients_list) >= 2:
            
            # Nom lisible de l'effet
            effect_label = _effect_label(effect)
            
            redundancies.append({
                'type': 'effect',  # Type de redondance
                'effect': effect,
                'effectLabel': effect_label,
                'count': len(ingredients_list),
                'ingredients': ingredients_list,
                'impact': 'medium',
                'recommendation': f"{effect_label} est traité par {len(ingredients_list)} actifs différents ({', '.join([i['displayName'] for i in ingredients_list])}). Un seul suffit généralement pour cet effet."
            })
    
    return redundancies


def _effect_label(effect: str) -> str:
    """
    Convertit un effet technique en label lisible
    
    Args:
        effect: Ex: 'anti_acne'
    
    Returns:
        Label (ex: 'Anti-acné')
    """
    labels = {
        'anti_acne': 'Anti-acné',
        'hydration': 'Hydratation',
        'brightening': 'Éclaircissement',
        'anti_aging': 'Anti-âge',
        'exfoliation': 'Exfoliation',
        'oil_control': 'Contrôle sébum',
        'pore_refining': 'Affinement pores',
        'plumping': 'Repulpant',
        'antioxidant': 'Antioxydant',
        'cell_renewal': 'Renouvellement cellulaire'
    }
    return labels.get(effect, effect.replace('_', ' ').capitalize())