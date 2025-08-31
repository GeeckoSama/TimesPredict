#!/usr/bin/env python3
"""
Test du modèle TimesFM réel avec fine-tuning
"""

import sys
from pathlib import Path

# Ajouter le chemin des modules
sys.path.append(str(Path(__file__).parent))

from loto_v2 import LotoV2CLI

def test_stats_with_sequences():
    """Recalculer les stats avec séquences historiques"""
    print("🔄 Recalcul des statistiques avec séquences historiques...")
    cli = LotoV2CLI()
    cli.run_stats_calculation()
    print("✅ Stats recalculées\n")

def test_real_finetuning():
    """Test du fine-tuning réel"""
    print("🚀 Test du fine-tuning TimesFM réel...")
    cli = LotoV2CLI()
    cli.run_finetuning()
    print("✅ Fine-tuning terminé\n")

def test_real_predictions():
    """Test de prédictions avec le modèle fine-tuné"""
    print("🎯 Test prédictions avec modèle fine-tuné...")
    cli = LotoV2CLI()
    
    # Plusieurs prédictions pour tester
    for i in range(3):
        print(f"\n--- Prédiction {i+1} ---")
        cli.run_single_prediction()
    
    print("✅ Prédictions testées\n")

def main():
    """Lance tous les tests du modèle réel"""
    print("=" * 60)
    print("🧪 TESTS MODÈLE TIMESFM RÉEL")
    print("=" * 60)
    print()
    
    # 1. Recalculer stats avec séquences
    test_stats_with_sequences()
    
    # 2. Fine-tuning réel
    test_real_finetuning()
    
    # 3. Test prédictions
    test_real_predictions()
    
    print("🎉 Tous les tests du modèle réel terminés!")

if __name__ == "__main__":
    main()