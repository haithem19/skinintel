# plugins/routine_builder.py
"""
Construit les routines matin et soir
"""
from typing import List, Dict, Any


# Ordre d'application des produits (layering)
LAYERING_ORDER = {
    'cleanser': 1,
    'toner': 2,
    'essence': 3,
    'serum': 4,
    'eye_cream': 5,
    'moisturizer': 6,
    'sunscreen': 7,
    'oil': 8
}


def build_routines(enriched_products: List[Dict[str, Any]], 
                   suggestions: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Construit les routines matin et soir
    
    Args:
        enriched_products: Produits de l'utilisateur
        suggestions: Suggestions generees (optionnel)
    
    Returns:
        {
            'morning': [...],
            'evening': [...]
        }
    """
    # Separe les produits par moment d'utilisation
    morning_products = []
    evening_products = []
    both_products = []  # Produits utilisables matin ET soir
    
    for product in enriched_products:
        
        # Determine quand utiliser ce produit
        times = _get_recommendation_times(product)
        
        if 'morning' in times and 'evening' in times:
            both_products.append(product)
        elif 'morning' in times:
            morning_products.append(product)
        elif 'evening' in times:
            evening_products.append(product)
        else:
            # Par defaut, utilisable matin et soir
            both_products.append(product)
    
    # Construit la routine matin
    morning_routine = _build_single_routine(
        morning_products + both_products,
        'morning',
        suggestions
    )
    
    # Construit la routine soir
    evening_routine = _build_single_routine(
        evening_products + both_products,
        'evening',
        suggestions
    )
    
    return {
        'morning': morning_routine,
        'evening': evening_routine
    }


def _get_recommendation_times(product: Dict) -> List[str]:
    """
    Determine quand utiliser un produit
    
    Returns:
        Liste : ['morning'], ['evening'], ou ['morning', 'evening']
    """
    times = []
    
    # Verifie dans enrichedData pour chaque actif
    for ingredient_name in product.get('activeIngredients', []):
        
        enriched = product.get('enrichedData', {}).get(ingredient_name, {})
        rec_times = enriched.get('recommendation_time', [])
        
        times.extend(rec_times)
    
    # Si un actif dit "evening" uniquement, tout le produit est soir uniquement
    if 'evening' in times and 'morning' not in times:
        return ['evening']
    
    # Si un actif dit "morning" uniquement
    if 'morning' in times and 'evening' not in times:
        return ['morning']
    
    # Sinon, utilisable matin et soir
    return ['morning', 'evening']


def _build_single_routine(products: List[Dict], 
                          time_of_day: str,
                          suggestions: Dict = None) -> List[Dict]:
    """
    Construit une routine (matin OU soir)
    
    Args:
        products: Produits a inclure
        time_of_day: 'morning' ou 'evening'
        suggestions: Suggestions a ajouter (optionnel)
    
    Returns:
        Liste ordonnee des etapes
    """
    routine_steps = []
    
    # Trie les produits par ordre d'application
    sorted_products = sorted(
        products,
        key=lambda p: LAYERING_ORDER.get(p.get('productType', 'serum'), 99)
    )
    
    # Ajoute chaque produit comme une etape
    for product in sorted_products:
        
        # Recupere le nom d'affichage (premier actif principal)
        display_name = product.get('productName', 'Produit')
        
        routine_steps.append({
            'order': len(routine_steps) + 1,
            'category': product.get('productType', 'serum'),
            'productId': product.get('userProductId'),
            'productName': display_name,
            'isOwned': True,
            'isSuggestion': False,
            'instructions': _get_instructions(product.get('productType'))
        })
    
    # Ajoute les suggestions si presentes
    if suggestions and suggestions.get('count', 0) > 0:
        for sug in suggestions['details']:
            
            # Filtre selon le moment de la journee
            if time_of_day == 'morning' and sug['productType'] == 'sunscreen':
                routine_steps.append({
                    'order': len(routine_steps) + 1,
                    'category': sug['productType'],
                    'productName': sug['keyIngredientLabel'],
                    'isOwned': False,
                    'isSuggestion': True,
                    'priority': sug.get('priority', 'high'),
                    'instructions': _get_instructions(sug['productType']),
                    'note': sug.get('note', sug.get('reason', ''))
                })
            
            elif sug['productType'] == 'serum':
                # Insere au bon endroit (apres toner, avant moisturizer)
                insert_at = _find_insert_position(routine_steps, 'serum')
                routine_steps.insert(insert_at, {
                    'order': insert_at + 1,
                    'category': sug['productType'],
                    'productName': sug['keyIngredientLabel'],
                    'isOwned': False,
                    'isSuggestion': True,
                    'priority': sug.get('priority', 'high'),
                    'instructions': _get_instructions(sug['productType']),
                    'note': sug.get('reason', '')
                })
    
    # Re-numerote les etapes
    for i, step in enumerate(routine_steps, 1):
        step['order'] = i
    
    return routine_steps


def _find_insert_position(routine: List[Dict], category: str) -> int:
    """
    Trouve ou inserer un produit selon sa categorie
    
    Returns:
        Position d'insertion
    """
    target_order = LAYERING_ORDER.get(category, 4)
    
    for i, step in enumerate(routine):
        step_order = LAYERING_ORDER.get(step.get('category', 'serum'), 99)
        if step_order > target_order:
            return i
    
    return len(routine)


def _get_instructions(product_type: str) -> str:
    """
    Retourne les instructions d'application selon le type de produit
    
    Args:
        product_type: 'cleanser', 'serum', etc.
    
    Returns:
        Instructions texte
    """
    instructions = {
        'cleanser': 'Masser sur visage humide, rincer a l\'eau tiede',
        'toner': 'Appliquer avec un coton ou tapoter avec les mains',
        'essence': 'Tapoter doucement sur le visage',
        'serum': 'Appliquer 2-3 gouttes sur peau seche',
        'eye_cream': 'Tapoter delicatement autour des yeux',
        'moisturizer': 'Quantite noisette sur visage et cou',
        'sunscreen': 'Quantite genereuse, reappliquer toutes les 2h',
        'oil': 'Quelques gouttes en derniere etape'
    }
    
    return instructions.get(product_type, 'Appliquer sur peau propre')