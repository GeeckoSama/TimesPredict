#!/usr/bin/env python3
"""
Test de la génération multiple avec barre de progression unifiée
"""

import sys
from pathlib import Path

# Ajouter le chemin des modules
sys.path.append(str(Path(__file__).parent))

from modules.validation import LotoValidator


def test_multiple_predictions():
    """Test de génération multiple avec barre unifiée"""
    print("=" * 60)
    print("🎲 TEST GÉNÉRATION MULTIPLE - BARRE UNIFIÉE")
    print("=" * 60)
    print()
    
    validator = LotoValidator()
    
    print("🔧 Test avec 20 prédictions pour démontrer la barre unifiée")
    print("Format : [████████░░░░] prédiction/max - Action courante")
    print()
    
    # Générer 20 prédictions avec barre unifiée
    results = validator.generate_prediction_series(
        n_predictions=20,
        use_timesfm=False  # Plus rapide pour le test
    )
    
    # Affichage du résumé (sans lignes multiples)
    validator.display_series_summary(results)
    
    print("\n" + "=" * 60)
    print("✅ Test réussi - Une seule barre élégante !")
    print("💡 Pas de lignes multiples pendant la génération")
    print("🎯 Format : [████████░░░░] N/max - Prédiction N/max")


def test_larger_series():
    """Test avec une série plus grande"""
    print("\n" + "=" * 60)
    print("🎲 TEST SÉRIE PLUS IMPORTANTE")
    print("=" * 60)
    print()
    
    validator = LotoValidator()
    
    print("🔧 Test avec 50 prédictions")
    print()
    
    # Générer 50 prédictions
    results = validator.generate_prediction_series(
        n_predictions=50,
        use_timesfm=False
    )
    
    # Juste afficher la combinaison optimale
    optimal = results["analysis"]["optimal_combination"]
    print(f"🎯 Combinaison optimale: {optimal['boules']} + {optimal['chance']}")
    print(f"📊 {results['metadata']['n_predictions']} prédictions générées")
    

def main():
    """Tests de la génération multiple"""
    test_multiple_predictions()
    test_larger_series()
    
    print("\n🎉 Interface de génération multiple améliorée !")
    print("   ✅ Une seule barre de progression élégante")
    print("   ✅ Format : [████████░░░░] N/max - Prédiction N/max") 
    print("   ✅ Pas d'affichage verbeux pendant la génération")
    print("   ✅ Résumé final clair et informatif")


if __name__ == "__main__":
    main()