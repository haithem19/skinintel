# test_conflict_checker.py
from plugins.conflict_checker import check_conflicts

# Données de test
products = [
    {
        'userProductId': 'prod_1',
        'activeIngredients': ['niacinamide'],
        'enrichedData': {
            'niacinamide': {
                'interactions': ['vitamin_c'],
                'display_name': 'Niacinamide'
            }
        }
    },
    {
        'userProductId': 'prod_2',
        'activeIngredients': ['vitamin_c'],
        'enrichedData': {
            'vitamin_c': {
                'interactions': ['niacinamide', 'retinol'],
                'display_name': 'Vitamin C'
            }
        }
    }
]

print("🧪 Test Conflicts")
print("-" * 50)

result = check_conflicts(products)

print(f"Conflits : {result['hasConflicts']}")
print(f"Nombre : {result['count']}")

for conf in result['details']:
    print(f"\n⚠️ {conf['ingredientA']} × {conf['ingredientB']}")
    print(f"   {conf['recommendation']}")