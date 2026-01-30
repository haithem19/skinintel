# test_utils.py
"""
Script de test pour vérifier les utils
"""
from plugins.db_utils import get_ingredients_data, get_ingredients_by_effect

print("🧪 Test 1 : Récupérer des actifs")
print("-" * 50)

# Test avec niacinamide et retinol
ingredients = get_ingredients_data(['niacinamide', 'retinol'])

for ing in ingredients:
    print(f"✅ {ing['display_name']}")
    print(f"   Interactions: {ing['interactions']}")
    print(f"   Effets: {ing['effects']}")
    print()

print("🧪 Test 2 : Trouver actifs anti-acné")
print("-" * 50)

anti_acne = get_ingredients_by_effect('anti_acne')

for ing in anti_acne:
    print(f"✅ {ing['display_name']} ({ing['ingredient_name']})")
    print(f"   Type: {ing['product_type']}")
    print()

print("✅ Tests terminés !")
