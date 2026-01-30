# plugins/file_utils.py
"""
Utilitaires pour gérer les fichiers JSON
"""
import json
import os
from typing import Dict, Any


def read_json(filepath: str) -> Dict[str, Any]:
    """
    Lit un fichier JSON
    
    Args:
        filepath: Chemin vers le fichier JSON
        
    Returns:
        Contenu du fichier en dictionnaire
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def write_json(filepath: str, data: Dict[str, Any]) -> None:
    """
    Écrit un dictionnaire dans un fichier JSON
    
    Args:
        filepath: Chemin de destination
        data: Données à écrire
    """
    # Crée le dossier parent si nécessaire
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def delete_file(filepath: str) -> None:
    """
    Supprime un fichier
    
    Args:
        filepath: Chemin du fichier à supprimer
    """
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"✅ Deleted: {filepath}")
    else:
        print(f"⚠️ File not found: {filepath}")