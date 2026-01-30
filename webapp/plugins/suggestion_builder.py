# plugins/suggestion_builder.py
"""
Suggère des produits manquants selon le profil utilisateur
"""
from typing import List, Dict, Any
from db_utils import get_ingredients_by_effect


def generate_suggestions(profile: Dict[str, Any], enriched_products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Génère des suggestions de produits manquants
    
    Args:
        profile: Profil utilisateur
            {
                'skinType': 'combination',
                'skinConcerns': ['acne', 'dehydration'],
                'primaryGoal': 'treat_acne'
            }
        enriched_products: Produits actuels de l'utilisateur
    
    Returns:
        {
            'count': nombre,
            'details': [liste des suggestions]
        }
    """
    suggestions = []
    
    # Récupère les actifs que l'utilisateur possède déjà
    user_ingredients = _get_all_user_ingredients(enriched_products)
    
    # Récupère les types de produits que l'utilisateur possède
    user_product_types = _get_all_user_product_types(enriched_products)
    
    # Pour chaque problème de peau
    for concern in profile.get('skinConcerns', []):
        
        # Trouve quel effet traite ce problème
        effect = _concern_to_effect(concern)
        
        if not effect:
            continue
        
        # Cherche dans la DB les actifs qui ont cet effet
        ingredients = get_ingredients_by_effect(effect)
        
        # Vérifie si l'utilisateur a déjà un actif pour ce problème
        has_solution = False
        for ing in ingredients:
            if ing['ingredient_name'] in user_ingredients:
                has_solution = True
                break
        
        # Si pas de solution → suggère un produit
        if not has_solution and len(ingredients) > 0:
            
            # Prend le premier actif trouvé (on pourrait améliorer la logique)
            suggested = ingredients[0]
            
            suggestions.append({
                'priority': 'high',
                'concern': concern,
                'concernLabel': _concern_label(concern),
                'productType': suggested['product_type'],
                'keyIngredient': suggested['ingredient_name'],
                'keyIngredientLabel': suggested['display_name'],
                'reason': f"Pour traiter {_concern_label(concern).lower()}",
                'benefits': suggested.get('effects', [])
            })
    
    # TOUJOURS suggérer SPF si retinol détecté
    if 'retinol' in user_ingredients and 'sunscreen' not in user_product_types:
        suggestions.append({
            'priority': 'critical',
            'concern': 'sun_protection',
            'concernLabel': 'Protection solaire',
            'productType': 'sunscreen',
            'keyIngredient': 'SPF 30+',
            'keyIngredientLabel': 'Crème solaire SPF 30+',
            'reason': 'OBLIGATOIRE avec rétinol',
            'note': '⚠️ Le rétinol rend la peau photosensible'
        })
    
    return {
        'count': len(suggestions),
        'details': suggestions
    }


def _get_all_user_ingredients(enriched_products: List[Dict]) -> List[str]:
    """
    Récupère tous les actifs que l'utilisateur possède
    
    Returns:
        Liste d'actifs (ex: ['niacinamide', 'retinol'])
    """
    ingredients = []
    for product in enriched_products:
        ingredients.extend(product.get('activeIngredients', []))
    return list(set(ingredients))  # Enlève les doublons


def _get_all_user_product_types(enriched_products: List[Dict]) -> List[str]:
    """
    Récupère tous les types de produits que l'utilisateur possède
    
    Returns:
        Liste de types (ex: ['serum', 'moisturizer'])
    """
    types = []
    for product in enriched_products:
        types.append(product.get('productType'))
    return list(set(types))


def _concern_to_effect(concern: str) -> str:
    """
    Convertit un problème de peau en effet recherché
    
    Args:
        concern: Ex: 'acne', 'dehydration'
    
    Returns:
        Effet correspondant (ex: 'anti_acne', 'hydration')
    """
    mapping = {
        'acne': 'anti_acne',
        'dark_spots': 'brightening',
        'aging': 'anti_aging',
        'dehydration': 'hydration',
        'dryness': 'hydration',
        'redness': 'anti_acne',
        'sensitivity': 'anti_acne',
        'dullness': 'brightening',
        'uneven_texture': 'exfoliation',
        'large_pores': 'pore_refining'
    }
    return mapping.get(concern, None)


def _concern_label(concern: str) -> str:
    """
    Convertit un concern technique en label lisible
    
    Args:
        concern: Ex: 'acne'
    
    Returns:
        Label (ex: 'Acné')
    """
    labels = {
        'acne': 'Acné',
        'dark_spots': 'Taches pigmentaires',
        'aging': 'Vieillissement',
        'dehydration': 'Déshydratation',
        'dryness': 'Sécheresse',
        'redness': 'Rougeurs',
        'sensitivity': 'Sensibilité',
        'dullness': 'Teint terne',
        'uneven_texture': 'Texture irrégulière',
        'large_pores': 'Pores dilatés'
    }
    if concern in labels:
        return labels[concern]
    else:
        return concern.replace('_', ' ').title()