# plugins/conflict_checker.py
"""
Détecte les conflits entre actifs cosmétiques
"""
from typing import List, Dict, Any


def check_conflicts(enriched_products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Détecte les conflits entre produits
    
    Logique simple :
    - Pour chaque paire de produits
    - Vérifie si un actif de A est dans les interactions de B
    
    Args:
        enriched_products: Produits avec données enrichies depuis la DB
    
    Returns:
        {
            'hasConflicts': True/False,
            'count': nombre,
            'details': [liste des conflits]
        }
    """
    conflicts = []
    
    # Compare chaque paire de produits (évite les doublons)
    for i in range(len(enriched_products)):
        for j in range(i + 1, len(enriched_products)):
            
            product_a = enriched_products[i]
            product_b = enriched_products[j]
            
            # Récupère tous les actifs et leurs interactions
            ingredients_a = product_a.get('activeIngredients', [])
            ingredients_b = product_b.get('activeIngredients', [])
            
            enriched_a = product_a.get('enrichedData', {})
            enriched_b = product_b.get('enrichedData', {})
            
            # Vérifie les conflits dans les 2 sens
            conflicts.extend(_check_pair(
                ingredients_a, enriched_a, product_a['userProductId'],
                ingredients_b, enriched_b, product_b['userProductId']
            ))
    
    return {
        'hasConflicts': len(conflicts) > 0,
        'count': len(conflicts),
        'details': conflicts
    }


def _check_pair(ingredients_a, enriched_a, product_id_a,
                ingredients_b, enriched_b, product_id_b):
    """
    Vérifie les conflits entre 2 produits
    
    Returns:
        Liste des conflits trouvés
    """
    conflicts = []
    
    # Pour chaque actif de A
    for ing_a in ingredients_a:
        
        # Récupère ses interactions depuis enrichedData
        if ing_a not in enriched_a:
            continue
        
        interactions = enriched_a[ing_a].get('interactions', [])
        display_a = enriched_a[ing_a].get('display_name', ing_a)
        
        # Vérifie si un actif de B est incompatible
        for ing_b in ingredients_b:
            
            if ing_b in interactions:
                
                # Conflit trouvé !
                display_b = enriched_b.get(ing_b, {}).get('display_name', ing_b)
                
                conflicts.append({
                    'severity': 'medium',  # On simplifie : tout est medium
                    'ingredientA': display_a,
                    'ingredientB': display_b,
                    'foundInProducts': [product_id_a, product_id_b],
                    'reason': f"{display_a} et {display_b} peuvent interagir négativement",
                    'recommendation': "Séparer l'utilisation (matin/soir ou jours alternés)"
                })
    
    return conflicts