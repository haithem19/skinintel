# test_suggestion_builder.py
# -*- coding: utf-8 -*-
from plugins.suggestion_builder import generate_suggestions

profile = {
    'skinType': 'combination',
    'skinConcerns': ['acne', 'dehydration']
}

products = [
    {
        'userProductId': 'prod_1',
        'productType': 'serum',
        'activeIngredients': ['retinol'],
        'enrichedData': {
            'retinol': {'display_name': 'Retinol'}
        }
    }
]

print("Test Suggestions")
print("-" * 50)

result = generate_suggestions(profile, products)

print("Suggestions:", result['count'])
print("")

for sug in result['details']:
    print("Suggestion:", sug['keyIngredientLabel'], "(" + sug['productType'] + ")")
    print("  Priorite:", sug['priority'])
    print("  Pour:", sug['concernLabel'])
    print("  Raison:", sug['reason'])
    if 'note' in sug:
        print("  ", sug['note'])
    print("")