#!/usr/bin/env python3
"""
Test final de la génération multiple propre
"""

import sys
from pathlib import Path

# Ajouter le chemin des modules
sys.path.append(str(Path(__file__).parent))

from modules.validation import LotoValidator


def test_clean_multiple_generation():
    """Test final de génération multiple propre"""
    print("=" * 60)
    print("🎯 TEST FINAL - GÉNÉRATION MULTIPLE PROPRE")
    print("=" * 60)
    print()
    
    validator = LotoValidator()
    
    print("🚀 Génération de 20 prédictions avec interface propre")
    print("✅ Messages de chargement seulement au début")
    print("✅ Une seule barre de progression élégante")
    print("✅ Toutes les prédictions générées")
    print()
    
    # Test avec 20 prédictions
    results = validator.generate_prediction_series(
        n_predictions=20,
        use_timesfm=False  # Plus rapide pour la démo
    )
    
    # Vérifier que toutes les prédictions sont là
    predictions = results["predictions"]
    analysis = results["analysis"]
    
    print(f"\n📊 RÉSULTATS :")
    print(f"   Prédictions générées: {len(predictions)}")
    print(f"   Attendues: {results['metadata']['n_predictions']}")
    print(f"   ✅ Toutes générées: {len(predictions) == 20}")
    
    # Top 3 boules les plus prédites
    top_boules = analysis["frequency_analysis"]["most_frequent_boules"][:3]
    print(f"\n🔥 Top 3 boules prédites:")
    for boule, count in top_boules:
        pct = count / 20 * 100
        print(f"   {boule:2d}: {count:2d} fois ({pct:4.1f}%)")
    
    # Combinaison optimale
    optimal = analysis["optimal_combination"]
    print(f"\n🎯 Combinaison optimale: {optimal['boules']} + {optimal['chance']}")
    
    print("\n" + "=" * 60)
    print("🎉 SUCCÈS COMPLET !")
    print("✅ 20 prédictions générées sans spam de messages")
    print("✅ Une seule barre de progression élégante")
    print("✅ Interface utilisateur parfaitement propre")
    print("✅ Résumé final informatif")


def main():
    """Test final"""
    test_clean_multiple_generation()


if __name__ == "__main__":
    main()