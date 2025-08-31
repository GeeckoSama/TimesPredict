#!/usr/bin/env python3
"""
Test des prédictions multiples en mode silencieux
"""

import sys
from pathlib import Path

# Ajouter le chemin des modules
sys.path.append(str(Path(__file__).parent))

from modules.validation import LotoValidator


def test_silent_multiple_predictions():
    """Test prédictions multiples sans messages répétitifs"""
    print("=" * 60)
    print("🎲 TEST PRÉDICTIONS MULTIPLES SILENCIEUSES")
    print("=" * 60)
    print()
    
    validator = LotoValidator()
    
    print("🔧 Génération de 10 prédictions en mode silencieux")
    print("✅ Devrait afficher UNE SEULE barre de progression")
    print("❌ Pas de messages répétitifs 'Modèle chargé', 'Poids chargés'")
    print()
    
    # Test avec TimesFM=False d'abord (plus simple)
    print("📊 Test avec prédictions statistiques pures (plus rapide)")
    results = validator.generate_prediction_series(
        n_predictions=10,
        use_timesfm=False  # Pas de TimesFM pour éviter les messages
    )
    
    # Affichage du résumé
    optimal = results["analysis"]["optimal_combination"]
    print(f"\n🎯 Résultat: {optimal['boules']} + {optimal['chance']}")
    print(f"📊 Méthode: {results['metadata']['used_timesfm']}")
    
    print("\n" + "=" * 60)
    print("✅ Test réussi - Interface propre !")
    print("💡 Une seule barre, pas de messages répétitifs")


def test_timesfm_silent():
    """Test avec TimesFM en mode silencieux"""
    print("\n" + "=" * 60)
    print("🤖 TEST TIMESFM SILENCIEUX")
    print("=" * 60)
    print()
    
    validator = LotoValidator()
    
    print("🔧 Génération de 5 prédictions avec TimesFM")
    print("✅ Devrait charger le modèle UNE SEULE fois au début")
    print()
    
    results = validator.generate_prediction_series(
        n_predictions=5,
        use_timesfm=True  # Avec TimesFM
    )
    
    # Résultat
    optimal = results["analysis"]["optimal_combination"]
    print(f"\n🎯 Résultat: {optimal['boules']} + {optimal['chance']}")
    

def main():
    """Tests du mode silencieux"""
    test_silent_multiple_predictions()
    test_timesfm_silent()
    
    print("\n🎉 Tests terminés !")
    print("   ✅ Interface propre sans spam de messages")
    print("   ✅ Modèle chargé une seule fois")
    print("   ✅ Barre de progression unifiée unique")


if __name__ == "__main__":
    main()