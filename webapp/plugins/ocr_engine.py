# plugins/ocr_engine.py
"""
OCR Engine for extracting ingredient text from product images.
Uses Tesseract OCR with OpenCV preprocessing for better accuracy.
"""
import pytesseract
import cv2
import numpy as np
import re
import os
from typing import Dict, List, Any, Optional
from .db_utils import get_db_connection


def _configure_tesseract():
    """Configure Tesseract path based on environment."""
    # Docker/Linux: tesseract is in PATH
    # Windows local dev: use explicit path
    if os.name == 'nt':  # Windows
        tesseract_path = os.environ.get(
            'TESSERACT_CMD',
            r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        )
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    # On Linux/Docker, tesseract should be in PATH


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better OCR accuracy.
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        Preprocessed grayscale image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better text extraction
    # This works better than simple Otsu for ingredient labels
    processed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    
    # Optional: denoise
    processed = cv2.medianBlur(processed, 3)
    
    return processed


def extract_text_from_image(image_path: str) -> str:
    """
    Extract raw text from an image using OCR.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text string
    """
    _configure_tesseract()
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Preprocess
    processed = preprocess_image(img)
    
    # OCR with custom config for ingredient lists
    # PSM 6 = Assume a single uniform block of text
    custom_config = r'--oem 3 --psm 6'
    raw_text = pytesseract.image_to_string(processed, config=custom_config)
    
    return raw_text


def extract_text_from_bytes(image_bytes: bytes) -> str:
    """
    Extract raw text from image bytes (for Flask file uploads).
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Extracted text string
    """
    _configure_tesseract()
    
    # Decode image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image from bytes")
    
    # Preprocess
    processed = preprocess_image(img)
    
    # OCR
    custom_config = r'--oem 3 --psm 6'
    raw_text = pytesseract.image_to_string(processed, config=custom_config)
    
    return raw_text


def clean_extracted_text(raw_text: str) -> str:
    """
    Clean and normalize OCR output.
    
    Args:
        raw_text: Raw OCR text
        
    Returns:
        Cleaned text
    """
    # Lowercase
    text = raw_text.lower()
    
    # Remove common OCR artifacts
    text = re.sub(r'[|}{[\]\\]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def get_known_ingredients_from_db() -> List[str]:
    """
    Fetch all known ingredient names from the database.
    
    Returns:
        List of ingredient names (lowercase)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Try active_ingredients_database first
        cursor.execute("""
            SELECT DISTINCT LOWER(ingredient_name) 
            FROM active_ingredients_database 
            WHERE ingredient_name IS NOT NULL
        """)
        ingredients = [row[0] for row in cursor.fetchall()]
        
        # Also check ewg_ingredients_dim if it exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'ewg_ingredients_dim'
            )
        """)
        if cursor.fetchone()[0]:
            cursor.execute("""
                SELECT DISTINCT LOWER(ingredient) 
                FROM ewg_ingredients_dim 
                WHERE ingredient IS NOT NULL
            """)
            ingredients.extend([row[0] for row in cursor.fetchall()])
        
        cursor.close()
        conn.close()
        
        # Deduplicate
        return list(set(ingredients))
        
    except Exception as e:
        print(f"⚠️ Database error fetching ingredients: {e}")
        return []


def match_ingredients(text: str, known_ingredients: Optional[List[str]] = None) -> List[str]:
    """
    Match extracted text against known ingredient names.
    
    Args:
        text: Cleaned OCR text
        known_ingredients: Optional list of known ingredients (fetched from DB if None)
        
    Returns:
        List of matched ingredient names
    """
    if known_ingredients is None:
        known_ingredients = get_known_ingredients_from_db()
    
    if not known_ingredients:
        return []
    
    text_lower = text.lower()
    
    # Find which known ingredients appear in the text
    matched = []
    for ingredient in known_ingredients:
        if ingredient and ingredient in text_lower:
            matched.append(ingredient)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_matched = []
    for ing in matched:
        if ing not in seen:
            seen.add(ing)
            unique_matched.append(ing)
    
    return unique_matched


def extract_ingredients_from_image(image_path: str) -> Dict[str, Any]:
    """
    Full pipeline: extract ingredients from a product image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with:
        - raw_text: Original OCR output (for debugging)
        - clean_text: Cleaned/normalized text
        - detected_actives: List of matched ingredient names
    """
    # 1. Extract raw text
    raw_text = extract_text_from_image(image_path)
    
    # 2. Clean text
    clean_text = clean_extracted_text(raw_text)
    
    # 3. Match against known ingredients
    detected_actives = match_ingredients(clean_text)
    
    return {
        "raw_text": raw_text,
        "clean_text": clean_text,
        "detected_actives": detected_actives
    }


def extract_ingredients_from_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """
    Full pipeline: extract ingredients from image bytes (Flask upload).
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Dictionary with raw_text, clean_text, detected_actives
    """
    # 1. Extract raw text
    raw_text = extract_text_from_bytes(image_bytes)
    
    # 2. Clean text
    clean_text = clean_extracted_text(raw_text)
    
    # 3. Match against known ingredients
    detected_actives = match_ingredients(clean_text)
    
    return {
        "raw_text": raw_text,
        "clean_text": clean_text,
        "detected_actives": detected_actives
    }


def process_multiple_images(image_paths: List[str]) -> Dict[str, Any]:
    """
    Process multiple images and combine results.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Combined results with all detected ingredients
    """
    all_raw_texts = []
    all_detected = set()
    
    for path in image_paths:
        try:
            result = extract_ingredients_from_image(path)
            all_raw_texts.append(result["raw_text"])
            all_detected.update(result["detected_actives"])
        except Exception as e:
            print(f"⚠️ Error processing {path}: {e}")
    
    return {
        "raw_texts": all_raw_texts,
        "detected_actives": list(all_detected)
    }


def process_multiple_image_bytes(images_bytes: List[bytes]) -> Dict[str, Any]:
    """
    Process multiple image byte arrays - each image is treated as a separate product.
    
    Args:
        images_bytes: List of raw image bytes (each = 1 product)
        
    Returns:
        Dictionary with:
        - products: List of products, each with its own ingredients
        - raw_texts: Raw OCR texts for debugging
        - detected_actives: Flat list of all unique ingredients (for backward compat)
    """
    all_raw_texts = []
    all_detected = set()
    products = []
    
    for i, img_bytes in enumerate(images_bytes):
        try:
            result = extract_ingredients_from_bytes(img_bytes)
            all_raw_texts.append(result["raw_text"])
            
            product_ingredients = result["detected_actives"]
            all_detected.update(product_ingredients)
            
            # Create a product entry for this image
            if product_ingredients:
                # Generate product name from top ingredients (max 3)
                top_ingredients = product_ingredients[:3]
                product_name = " + ".join([ing.title() for ing in top_ingredients])
                if len(product_ingredients) > 3:
                    product_name += f" (+{len(product_ingredients) - 3} more)"
                
                products.append({
                    "productId": f"scanned_product_{i+1}",
                    "productName": product_name,
                    "ingredients": product_ingredients,
                    "imageIndex": i
                })
            
        except Exception as e:
            print(f"⚠️ Error processing image {i+1}: {e}")
    
    return {
        "raw_texts": all_raw_texts,
        "detected_actives": list(all_detected),
        "products": products  # NEW: products grouped by image
    }
