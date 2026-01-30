from flask import Flask, render_template, request, redirect, url_for, session
import requests
import json
import time
import os
from datetime import datetime
from werkzeug.utils import secure_filename

# OCR Engine import
from plugins.ocr_engine import extract_ingredients_from_bytes, process_multiple_image_bytes

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'skintel_secret_key_2026')

# Upload configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _data_path(*parts: str) -> str:
    base = os.environ.get("APP_DATA_DIR", os.path.join(os.getcwd(), "data"))
    return os.path.join(base, *parts)


def trigger_airflow_analysis(user_id, profile_data, products_data):
    """
    Déclenche l'analyse Airflow via API REST
    Crée le fichier input dans un volume partagé, appelle l'API, et attend les résultats
    """
    try:
        # Création du fichier JSON input
        input_data = {
            'userId': user_id,
            'profile': profile_data,
            'userProducts': products_data
        }
        
        filename = f'{user_id}_input.json'
        temp_dir = _data_path('temp')
        os.makedirs(temp_dir, exist_ok=True)
        input_path = os.path.join(temp_dir, filename)

        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(input_data, f, indent=2)
        
        print(f"Fichier input écrit : {input_path}")
        
        # Appel API Airflow pour déclencher le DAG
        print(f"Déclenchement du DAG pour {user_id}...")
        
        airflow_url = os.environ.get('AIRFLOW_URL', 'http://localhost:8080').rstrip('/')
        airflow_user = os.environ.get('AIRFLOW_USER', 'admin')
        airflow_password = os.environ.get('AIRFLOW_PASSWORD', 'admin')

        response = requests.post(
            f'{airflow_url}/api/v1/dags/skincare_user_analysis/dagRuns',
            auth=(airflow_user, airflow_password),
            headers={'Content-Type': 'application/json'},
            json={'conf': {'user_id': user_id}}
        )
        
        print(f"Réponse API Airflow - Status: {response.status_code}")
        
        if response.status_code not in [200, 201]:
            print(f"Erreur API : {response.text}")
            return None
        
        print("DAG déclenché avec succès, attente des résultats...")
        
        # Polling : attente du fichier résultat (max 60 secondes)
        results_dir = _data_path('results')
        os.makedirs(results_dir, exist_ok=True)
        result_path = os.path.join(results_dir, f'{user_id}_results.json')

        for attempt in range(30):
            if os.path.exists(result_path):
                with open(result_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"Résultats récupérés pour {user_id}")
                return results

            time.sleep(2)
        
        # Timeout après 60 secondes
        print(f"Timeout : résultats non disponibles après 60 secondes")
        return None
        
    except Exception as e:
        print(f"Exception dans trigger_airflow_analysis : {str(e)}")
        return None


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        images = request.files.getlist("ingredients_images")
        print(f"{len(images)} image(s) uploadée(s)")

        if not images or images[0].filename == "":
            return "No images uploaded", 400

        # Filter valid images
        valid_images = [img for img in images if img and allowed_file(img.filename)]
        
        if not valid_images:
            return "No valid images uploaded", 400

        # Process images with OCR - each image = 1 product
        try:
            images_bytes = [img.read() for img in valid_images]
            ocr_results = process_multiple_image_bytes(images_bytes)
            
            detected_ingredients = ocr_results.get("detected_actives", [])
            scanned_products = ocr_results.get("products", [])
            raw_texts = ocr_results.get("raw_texts", [])
            
            print(f"OCR detected {len(scanned_products)} products with {len(detected_ingredients)} total ingredients")
            for prod in scanned_products:
                print(f"  - {prod['productName']}: {prod['ingredients']}")
            
            # Store in session for use in form
            session['detected_ingredients'] = detected_ingredients
            session['scanned_products'] = scanned_products  # NEW: products grouped by image
            session['ocr_raw_texts'] = raw_texts
            
        except Exception as e:
            print(f"OCR error: {e}")
            session['detected_ingredients'] = []
            session['scanned_products'] = []
            session['ocr_error'] = str(e)

        return redirect(url_for("form"))

    return render_template("upload.html")


@app.route("/form", methods=["GET", "POST"])
def form():
    # Retrieve OCR results from session (if coming from upload)
    detected_ingredients = session.get('detected_ingredients', [])
    scanned_products = session.get('scanned_products', [])
    ocr_error = session.pop('ocr_error', None)
    
    if request.method == "POST":
        # Récupération des données du formulaire
        skin_type = request.form.get("skinType")
        skin_concerns = request.form.getlist("skinConcerns")
        primary_goal = request.form.get("primaryGoal", "general")

        profile_data = {
            "skinType": skin_type,
            "skinConcerns": skin_concerns,
            "primaryGoal": primary_goal
        }

        # Génération d'un ID unique pour l'utilisateur
        user_id = f"user_{int(datetime.now().timestamp())}"
        
        # Get scanned products from OCR (each image = 1 product)
        scanned_products = session.pop('scanned_products', [])
        manual_ingredients = request.form.getlist("manualIngredients")
        
        # Build products_data from scanned products (grouped by image)
        products_data = []
        
        if scanned_products:
            for prod in scanned_products:
                products_data.append({
                    "userProductId": prod["productId"],
                    "productName": prod["productName"],
                    "productType": "serum",  # Default type
                    "activeIngredients": prod["ingredients"]
                })
        
        # Add manual ingredients as a separate product if provided
        if manual_ingredients:
            products_data.append({
                "userProductId": "manual_product_001",
                "productName": "Manually Added Ingredients",
                "productType": "serum",
                "activeIngredients": manual_ingredients
            })
        
        # Fallback: dummy products if nothing detected
        if not products_data:
            products_data = [
                {
                    "userProductId": "dummy_prod_001",
                    "productName": "Test Niacinamide Serum",
                    "productType": "serum",
                    "activeIngredients": ["niacinamide"]
                },
                {
                    "userProductId": "dummy_prod_002",
                    "productName": "Test Retinol Cream",
                    "productType": "serum",
                    "activeIngredients": ["retinol"]
                }
            ]
        
        print(f"Analyse lancée pour {user_id} avec {len(products_data)} produits")
        
        # Déclenchement de l'analyse Airflow
        results = trigger_airflow_analysis(user_id, profile_data, products_data)
        
        if results:
            # Stockage des résultats dans la session
            session['results'] = results
            session['user_id'] = user_id
            return redirect(url_for("result"))
        else:
            # Timeout ou erreur
            session['error'] = "L'analyse a pris trop de temps ou a échoué"
            return redirect(url_for("result"))

    return render_template(
        "form.html",
        detected_ingredients=detected_ingredients,
        scanned_products=scanned_products,
        ocr_error=ocr_error
    )


@app.route("/result")
def result():
    # Récupération des résultats depuis la session
    results = session.pop('results', None)
    error = session.pop('error', None)
    user_id = session.pop('user_id', None)
    
    return render_template("result.html", results=results, error=error, user_id=user_id)


if __name__ == "__main__":
    print("Démarrage de l'application Flask...")
    app.run(debug=True)