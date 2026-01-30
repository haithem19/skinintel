-- ============================================================================
-- BASE DE DONNÉES DES ACTIFS COSMÉTIQUES
-- ============================================================================

-- Table principale : Liste des actifs cosmétiques
CREATE TABLE IF NOT EXISTS active_ingredients_database (
    -- Identifiant unique de l'actif
    ingredient_id VARCHAR(50) PRIMARY KEY,
    
    -- Nom de l'actif (en minuscules, ex: "niacinamide")
    ingredient_name VARCHAR(100) UNIQUE NOT NULL,
    
    -- Nom à afficher (ex: "Niacinamide")
    display_name VARCHAR(100) NOT NULL,
    
    -- Type de produit recommandé (ex: "serum", "cream", "all")
    product_type VARCHAR(50),
    
    -- Types de peau compatibles (array)
    -- Ex: ['oily', 'combination', 'all']
    skin_type_compatibility TEXT[],
    
    -- Liste des actifs incompatibles (array)
    -- Ex: ['vitamin_c', 'retinol']
    interactions TEXT[],
    
    -- Moment d'utilisation (array)
    -- Ex: ['morning', 'evening'] ou ['morning_evening']
    recommendation_time TEXT[],
    
    -- Effets de l'actif (array)
    -- Ex: ['anti_acne', 'hydration', 'brightening']
    effects TEXT[]
);

-- Index pour recherche rapide par nom
CREATE INDEX idx_ingredient_name ON active_ingredients_database(ingredient_name);

-- ============================================================================
-- DONNÉES DE TEST (5 actifs)
-- ============================================================================

INSERT INTO active_ingredients_database VALUES
-- NIACINAMIDE
('niacinamide', 'niacinamide', 'Niacinamide',
 'serum',
 ARRAY['oily', 'combination', 'normal', 'all'],
 ARRAY['vitamin_c'],
 ARRAY['morning', 'evening'],
 ARRAY['anti_acne', 'oil_control', 'pore_refining']
),

-- RETINOL
('retinol', 'retinol', 'Retinol',
 'serum',
 ARRAY['normal', 'oily', 'combination', 'all'],
 ARRAY['vitamin_c', 'glycolic_acid', 'salicylic_acid'],
 ARRAY['evening'],
 ARRAY['anti_aging', 'cell_renewal']
),

-- HYALURONIC ACID
('hyaluronic_acid', 'hyaluronic_acid', 'Hyaluronic Acid',
 'serum',
 ARRAY['all'],
 ARRAY[]::TEXT[],
 ARRAY['morning', 'evening'],
 ARRAY['hydration', 'plumping']
),

-- VITAMIN C
('vitamin_c', 'vitamin_c', 'Vitamin C',
 'serum',
 ARRAY['all'],
 ARRAY['retinol', 'niacinamide'],
 ARRAY['morning'],
 ARRAY['brightening', 'antioxidant']
),

-- SALICYLIC ACID
('salicylic_acid', 'salicylic_acid', 'Salicylic Acid',
 'serum',
 ARRAY['oily', 'acne_prone'],
 ARRAY['retinol'],
 ARRAY['evening'],
 ARRAY['exfoliation', 'anti_acne']
);

-- Message de confirmation
DO $$
BEGIN
    RAISE NOTICE '✅ Base initialisée avec 5 actifs de test';
END $$;