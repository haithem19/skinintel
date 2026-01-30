# dags/user_analysis_dag.py
"""
DAG Airflow pour analyser les routines skincare des utilisateurs

Workflow :
1. Reçoit user_id via l'API REST
2. Lit les donnees utilisateur (profile + products)
3. Enrichit avec la base de donnees
4. Analyse : conflits, redondances, suggestions, routines
5. Sauvegarde les resultats
6. Nettoie les fichiers temporaires
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import json
import os

# Import de nos modules
import sys
sys.path.insert(0, '/opt/airflow/plugins')

from file_utils import read_json, write_json, delete_file
from db_utils import get_ingredients_data, get_normalized_ingredients_data, find_matching_products
from conflict_checker import check_conflicts
from redundancy_checker import check_redundancy
from suggestion_builder import generate_suggestions
from routine_builder import build_routines

# Configuration du DAG
default_args = {
    'owner': 'skincare-backend',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    'skincare_user_analysis',
    default_args=default_args,
    description='Analyse complete routine skincare utilisateur',
    schedule=None,  # Déclenchement uniquement via API
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['skincare', 'analysis']
)


def get_user_id(**context):
    """
    Récupère le user_id depuis la configuration du DAG Run (API)
    """
    dag_run = context.get('dag_run')
    
    if dag_run and dag_run.conf:
        user_id = dag_run.conf.get('user_id')
        if user_id:
            print(f"✅ User ID recu via API : {user_id}")
            context['task_instance'].xcom_push(key='user_id', value=user_id)
            return user_id
    
    raise ValueError("❌ Aucun user_id fourni via API")


def load_user_data(**context):
    """
    Charge les donnees utilisateur depuis les fichiers JSON
    """
    user_id = context['task_instance'].xcom_pull(key='user_id')
    input_file = f'/opt/airflow/data/temp/{user_id}_input.json'
    
    data = read_json(input_file)
    
    print(f"Donnees chargees pour {user_id}")
    print(f"  - Skin type: {data['profile']['skinType']}")
    print(f"  - Concerns: {data['profile']['skinConcerns']}")
    print(f"  - Products: {len(data['userProducts'])}")
    
    context['task_instance'].xcom_push(key='user_data', value=data)
    
    return data


def enrich_products(**context):
    """
    Enrichit les produits utilisateur avec les donnees de la base.
    - Normalise les noms d'ingredients OCR pour matcher la DB
    - Cherche des produits correspondants dans products_dim
    - Filtre pour ne garder que les ingredients trouvés dans la DB
    """
    user_data = context['task_instance'].xcom_pull(key='user_data')
    user_products = user_data['userProducts']
    
    enriched_products = []
    
    for product in user_products:
        raw_ingredients = product['activeIngredients']
        
        # Use normalized lookup to match OCR text to DB
        ingredients_data, matched_names = get_normalized_ingredients_data(raw_ingredients)
        
        # Skip products with no matching ingredients in DB
        if not ingredients_data:
            print(f"  ⚠️ No DB match for product: {product.get('productName', 'Unknown')}")
            print(f"     Raw ingredients: {raw_ingredients[:5]}...")
            continue
        
        enriched_data = {}
        for ing_data in ingredients_data:
            enriched_data[ing_data['ingredient_name']] = ing_data
        
        # Try to find matching product from products_dim
        product_name = product.get('productName', 'Produit')
        matching_products = []
        
        if product_name.startswith('Scanned Product') or 'Scanned' in product_name:
            # Try to find real product name from products_dim
            matching_products = find_matching_products(raw_ingredients, top_n=1)
            if matching_products:
                best_match = matching_products[0]
                product_name = f"{best_match['product_name']} ({best_match['company']})"
                print(f"  ✅ Matched to: {product_name} ({best_match['match_pct']}% match)")
            else:
                # Fallback: use first ingredient's display name
                product_name = ingredients_data[0].get('display_name', product_name)
        
        enriched_product = {
            **product,
            'productName': product_name,
            'enrichedData': enriched_data,
            'matchedProducts': matching_products,
            'matchedIngredientCount': len(ingredients_data),
            'totalScannedIngredients': len(raw_ingredients)
        }
        
        enriched_products.append(enriched_product)
    
    print(f"Produits enrichis: {len(enriched_products)} (sur {len(user_products)} scannes)")
    
    context['task_instance'].xcom_push(key='enriched_products', value=enriched_products)
    
    return enriched_products


def analyze_conflicts(**context):
    """
    Detecte les conflits entre actifs
    """
    enriched_products = context['task_instance'].xcom_pull(key='enriched_products')
    conflicts = check_conflicts(enriched_products)
    
    print(f"Conflits detectes: {conflicts['count']}")
    
    context['task_instance'].xcom_push(key='conflicts', value=conflicts)
    
    return conflicts


def analyze_redundancy(**context):
    """
    Detecte les redondances (actifs en double)
    """
    enriched_products = context['task_instance'].xcom_pull(key='enriched_products')
    redundancy = check_redundancy(enriched_products)
    
    print(f"Redondances detectees: {redundancy['count']}")
    
    context['task_instance'].xcom_push(key='redundancy', value=redundancy)
    
    return redundancy


def generate_suggestions_task(**context):
    """
    Genere les suggestions de produits manquants
    """
    user_data = context['task_instance'].xcom_pull(key='user_data')
    enriched_products = context['task_instance'].xcom_pull(key='enriched_products')
    
    profile = user_data['profile']
    suggestions = generate_suggestions(profile, enriched_products)
    
    print(f"Suggestions generees: {suggestions['count']}")
    
    context['task_instance'].xcom_push(key='suggestions', value=suggestions)
    
    return suggestions


def build_routines_task(**context):
    """
    Construit les routines matin et soir
    """
    enriched_products = context['task_instance'].xcom_pull(key='enriched_products')
    suggestions = context['task_instance'].xcom_pull(key='suggestions')
    
    routines = build_routines(enriched_products, suggestions)
    
    print(f"Routine matin: {len(routines['morning'])} etapes")
    print(f"Routine soir: {len(routines['evening'])} etapes")
    
    context['task_instance'].xcom_push(key='routines', value=routines)
    
    return routines


def save_results(**context):
    """
    Sauvegarde les resultats dans un fichier JSON
    """
    user_id = context['task_instance'].xcom_pull(key='user_id')
    
    conflicts = context['task_instance'].xcom_pull(key='conflicts')
    redundancy = context['task_instance'].xcom_pull(key='redundancy')
    suggestions = context['task_instance'].xcom_pull(key='suggestions')
    routines = context['task_instance'].xcom_pull(key='routines')
    
    results = {
        'analysisId': f'analysis_{user_id}_{int(datetime.now().timestamp())}',
        'userId': user_id,
        'timestamp': datetime.now().isoformat(),
        'conflicts': conflicts,
        'redundancy': redundancy,
        'suggestions': suggestions,
        'routines': routines
    }
    
    output_file = f'/opt/airflow/data/results/{user_id}_results.json'
    write_json(output_file, results)
    
    print(f"Resultats sauvegardes: {output_file}")
    
    return output_file


def cleanup(**context):
    """
    Nettoie les fichiers temporaires
    """
    user_id = context['task_instance'].xcom_pull(key='user_id')
    
    input_file = f'/opt/airflow/data/temp/{user_id}_input.json'
    delete_file(input_file)
    
    print("Nettoyage termine")


# ========================================
# DEFINITION DES TASKS
# ========================================

task_get_user = PythonOperator(
    task_id='get_user_id',
    python_callable=get_user_id,
    dag=dag
)

task_load = PythonOperator(
    task_id='load_user_data',
    python_callable=load_user_data,
    dag=dag
)

task_enrich = PythonOperator(
    task_id='enrich_products',
    python_callable=enrich_products,
    dag=dag
)

task_conflicts = PythonOperator(
    task_id='analyze_conflicts',
    python_callable=analyze_conflicts,
    dag=dag
)

task_redundancy = PythonOperator(
    task_id='analyze_redundancy',
    python_callable=analyze_redundancy,
    dag=dag
)

task_suggestions = PythonOperator(
    task_id='generate_suggestions',
    python_callable=generate_suggestions_task,
    dag=dag
)

task_routines = PythonOperator(
    task_id='build_routines',
    python_callable=build_routines_task,
    dag=dag
)

task_save = PythonOperator(
    task_id='save_results',
    python_callable=save_results,
    dag=dag
)

task_cleanup = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup,
    dag=dag
)

# ========================================
# DEFINITION DU WORKFLOW
# ========================================

task_get_user >> task_load >> task_enrich
task_enrich >> [task_conflicts, task_redundancy, task_suggestions]
[task_conflicts, task_redundancy, task_suggestions] >> task_routines
task_routines >> task_save >> task_cleanup