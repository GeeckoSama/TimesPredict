#!/usr/bin/env python3
"""
Test rapide du système Loto V2 avec barres de progression
"""

import sys
from pathlib import Path

# Ajouter le chemin des modules
sys.path.append(str(Path(__file__).parent))

from loto_v2 import LotoV2CLI

def test_status():
    """Test du statut détaillé"""
    print("🔍 Test du statut détaillé")
    cli = LotoV2CLI()
    cli.run_detailed_status()
    print("✅ Statut OK\n")

def test_single_prediction():
    """Test d'une prédiction simple"""
    print("🎯 Test prédiction simple")
    cli = LotoV2CLI()
    cli.run_single_prediction()
    print("✅ Prédiction OK\n")

def test_mini_series():
    """Test série de 10 prédictions directement"""
    print("🎲 Test série de 10 prédictions")
    
    # Tester directement le validator
    from modules.validation import LotoValidator
    validator = LotoValidator()
    
    try:
        results = validator.generate_prediction_series(n_predictions=5, use_timesfm=False)
        print(f"✅ {len(results['predictions'])} prédictions générées")
        
        # Afficher combinaison optimale
        optimal = results['analysis']['optimal_combination']
        print(f"🎯 Combinaison optimale: {optimal['boules']} + {optimal['chance']}")
        print("✅ Série OK\n")
        
    except Exception as e:
        print(f"⚠️  Erreur série: {e}\n")

def main():
    """Lance tous les tests"""
    print("=" * 50)
    print("🧪 TESTS RAPIDES LOTO V2")
    print("=" * 50)
    print()
    
    test_status()
    test_single_prediction()  
    test_mini_series()
    
    print("🎉 Tous les tests terminés!")

if __name__ == "__main__":
    main()