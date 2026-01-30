# test_routine_builder.py
from plugins.routine_builder import build_routines

# Produits de test
products = [
    {
        'userProductId': 'prod_1',
        'productName': 'CeraVe Cleanser',
        'productType': 'cleanser',
        'activeIngredients': ['ceramides'],
        'enrichedData': {
            'ceramides': {
                'recommendation_time': ['morning', 'evening']
            }
        }
    },
    {
        'userProductId': 'prod_2',
        'productName': 'Niacinamide Serum',
        'productType': 'serum',
        'activeIngredients': ['niacinamide'],
        'enrichedData': {
            'niacinamide': {
                'recommendation_time': ['morning', 'evening']
            }
        }
    },
    {
        'userProductId': 'prod_3',
        'productName': 'Retinol Serum',
        'productType': 'serum',
        'activeIngredients': ['retinol'],
        'enrichedData': {
            'retinol': {
                'recommendation_time': ['evening']
            }
        }
    }
]

# Suggestions
suggestions = {
    'count': 1,
    'details': [
        {
            'productType': 'sunscreen',
            'keyIngredientLabel': 'SPF 30+',
            'priority': 'critical',
            'note': 'OBLIGATOIRE avec retinol'
        }
    ]
}

print("Test Routines")
print("=" * 50)

result = build_routines(products, suggestions)

print("\nMORNING ROUTINE:")
print("-" * 30)
for step in result['morning']:
    print(f"{step['order']}. {step['productName']} ({step['category']})")
    if step.get('isSuggestion'):
        print(f"   [SUGGESTION] {step.get('note', '')}")

print("\nEVENING ROUTINE:")
print("-" * 30)
for step in result['evening']:
    print(f"{step['order']}. {step['productName']} ({step['category']})")
    if step.get('isSuggestion'):
        print(f"   [SUGGESTION] {step.get('note', '')}")