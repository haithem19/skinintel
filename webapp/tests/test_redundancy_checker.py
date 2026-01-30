# test_redundancy_advanced.py
from plugins.redundancy_checker import check_redundancy

# Données de test avec 2 types de redondances
products = [
    {
        'userProductId': 'prod_1',
        'productName': 'Niacinamide Serum 1',
        'activeIngredients': ['niacinamide'],
        'enrichedData': {
            'niacinamide': {
                'display_name': 'Niacinamide',
                'effects': ['anti_acne', 'oil_control']
            }
        }
    },
    {
        'userProductId': 'prod_2',
        'productName': 'Niacinamide Serum 2',
        'activeIngredients': ['niacinamide'],
        'enrichedData': {
            'niacinamide': {
                'display_name': 'Niacinamide',
                'effects': ['anti_acne', 'oil_control']
            }
        }
    },
    {
        'userProductId': 'prod_3',
        'productName': 'Salicylic Acid Toner',
        'activeIngredients': ['salicylic_acid'],
        'enrichedData': {
            'salicylic_acid': {
                'display_name': 'Salicylic Acid',
                'effects': ['anti_acne', 'exfoliation']
            }
        }
    },
    {
        'userProductId': 'prod_4',
        'productName': 'HA Serum',
        'activeIngredients': ['hyaluronic_acid'],
        'enrichedData': {
            'hyaluronic_acid': {
                'display_name': 'Hyaluronic Acid',
                'effects': ['hydration', 'plumping']
            }
        }
    },
    {
        'userProductId': 'prod_5',
        'productName': 'Moisturizer',
        'activeIngredients': ['glycerin'],
        'enrichedData': {
            'glycerin': {
                'display_name': 'Glycerin',
                'effects': ['hydration']
            }
        }
    }
]

print("Test Redundancy (Advanced)")
print("=" * 60)

result = check_redundancy(products)

print(f"\nRedondances detectees : {result['hasRedundancy']}")
print(f"Nombre total : {result['count']}")

print("\n--- TYPE 1 : Redondances d'ACTIFS ---")
for red in result['byIngredient']:
    print(f"\n{red['displayName']}")
    print(f"  Present dans {red['count']} produits :")
    for prod in red['foundInProducts']:
        print(f"    - {prod['productName']}")
    print(f"  Recommandation : {red['recommendation']}")

print("\n--- TYPE 2 : Redondances d'EFFETS ---")
for red in result['byEffect']:
    print(f"\n{red['effectLabel']}")
    print(f"  Traite par {red['count']} actifs :")
    for ing in red['ingredients']:
        print(f"    - {ing['displayName']} (dans {ing['productName']})")
    print(f"  Recommandation : {red['recommendation']}")