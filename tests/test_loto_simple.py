#!/usr/bin/env python3
"""
Test simple du système loto
"""

import sys
import os
sys.path.append("src")

print("🎰 TEST SIMPLE DU SYSTÈME LOTO")
print("=" * 50)

# Test des imports
try:
    from loto_predict.data.loto_data_processor import LotoDataProcessor
    print("✅ LotoDataProcessor importé")
except Exception as e:
    print(f"❌ LotoDataProcessor: {e}")

try:
    from loto_predict.models.multi_timesfm_predictor import MultiTimesFMPredictor
    print("✅ MultiTimesFMPredictor importé")
except Exception as e:
    print(f"❌ MultiTimesFMPredictor: {e}")

try:
    from loto_predict.analysis.loto_stat_analyzer import LotoStatAnalyzer
    print("✅ LotoStatAnalyzer importé")
except Exception as e:
    print(f"❌ LotoStatAnalyzer: {e}")

try:
    from loto_predict.optimization.combination_generator import CombinationGenerator
    print("✅ CombinationGenerator importé")
except Exception as e:
    print(f"❌ CombinationGenerator: {e}")

try:
    from loto_predict.validation.backtest_validator import BacktestValidator
    print("✅ BacktestValidator importé")
except Exception as e:
    print(f"❌ BacktestValidator: {e}")

print("\n🔍 Recherche du fichier de données loto...")
fichiers_possibles = [
    "data/raw/loto_201911.csv",
    "data/raw/loto.csv", 
    "loto_201911.csv",
    "loto.csv"
]

fichier_trouve = None
for fichier in fichiers_possibles:
    if os.path.exists(fichier):
        print(f"✅ Fichier trouvé: {fichier}")
        fichier_trouve = fichier
        break

if not fichier_trouve:
    print("❌ Aucun fichier de données loto trouvé")
    print("Fichiers recherchés:", fichiers_possibles)
else:
    print(f"\n📊 Test de chargement des données: {fichier_trouve}")
    try:
        data_processor = LotoDataProcessor(fichier_trouve)
        raw_data = data_processor.load_data()
        print(f"✅ Données chargées: {len(raw_data)} tirages")
        
        print("\n🤖 Test du prédicteur en mode simulation...")
        predictor = MultiTimesFMPredictor(
            model_repo="google/timesfm-2.0-500m-pytorch",
            backend="cpu", 
            horizon_len=1
        )
        
        success = predictor.load_models(simulation_mode=True)
        if success:
            print("✅ Modèles de simulation chargés")
            
            print("\n🎯 Test de prédiction...")
            # Traiter les données d'abord
            processed_result = data_processor.process_data()
            time_series = data_processor.create_time_series()
            prediction = predictor.predict_next_combination(time_series)
            
            print(f"✅ Prédiction générée:")
            print(f"   Boules: {prediction['combination']['boules']}")
            print(f"   Chance: {prediction['combination']['numero_chance']}")
        else:
            print("❌ Échec du chargement des modèles de simulation")
            
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()

print("\n🎉 Test terminé !")