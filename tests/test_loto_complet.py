#!/usr/bin/env python3
"""
Test complet du système de prédiction loto avec le dataset fusionné
"""

import sys
sys.path.append("src")

import pandas as pd
from loto_predict.models.multi_timesfm_predictor import MultiTimesFMPredictor

def test_système_complet():
    print("🎰 TEST COMPLET - SYSTÈME LOTO TIMESFM")
    print("=" * 50)
    
    # 1. Chargement des données fusionnées
    print("📂 Chargement des données fusionnées...")
    try:
        df = pd.read_csv("data/raw/loto_complet_fusionne.csv", sep=';')
        print(f"✅ {len(df)} tirages chargés")
        print(f"   Période: {df['date_de_tirage'].iloc[0]} → {df['date_de_tirage'].iloc[-1]}")
    except Exception as e:
        print(f"❌ Erreur chargement données: {e}")
        return
    
    # 2. Initialisation du prédicteur multi-modèle
    print("\n🤖 Initialisation du prédicteur multi-TimesFM...")
    try:
        predictor = MultiTimesFMPredictor(
            horizon_len=1,
            backend="cpu",
            model_repo="google/timesfm-1.0-200m"  # Modèle plus léger pour le test
        )
        print("✅ Prédicteur créé")
    except Exception as e:
        print(f"❌ Erreur création prédicteur: {e}")
        return
    
    # 3. Chargement des modèles
    print("\n🔄 Chargement des modèles TimesFM...")
    try:
        success = predictor.load_models(simulation_mode=False)
        print(f"✅ Modèles chargés: {success}")
    except Exception as e:
        print(f"❌ Erreur chargement modèles: {e}")
        return
    
    # 4. Génération de prédictions
    print("\n🎯 Génération des prédictions...")
    try:
        # Utiliser les derniers 100 tirages comme contexte
        context_data = df.tail(100)
        
        # Préparer les données de séries temporelles
        series_data = {}
        for component in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'numero_chance']:
            series_data[component] = context_data[component].values
        
        predictions = predictor.predict_next_combination(series_data)
        
        print(f"✅ Prédictions générées")
        
        # Afficher les résultats
        print(f"\n🎲 Prédiction:")
        print(f"   Clés disponibles: {list(predictions.keys())}")
        
        # Extraire les boules selon le format retourné
        if 'final_combination' in predictions:
            final_combo = predictions['final_combination']
            boules = final_combo['boules']
            chance = final_combo['numero_chance']
        elif 'combination' in predictions:
            combo = predictions['combination']
            boules = combo['boules'] if 'boules' in combo else [combo.get(f'boule_{j}', 0) for j in range(1, 6)]
            chance = combo.get('numero_chance', 0)
        else:
            # Format individuel
            boules = [predictions.get(f'boule_{j}', 0) for j in range(1, 6)]
            chance = predictions.get('numero_chance', 0)
            
        print(f"   Boules: {boules}")
        print(f"   Chance: {chance}")
        
        if 'confidence_moyenne' in predictions:
            print(f"   Confiance: {predictions['confidence_moyenne']:.1%}")
    
    except Exception as e:
        print(f"❌ Erreur génération prédictions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎉 TEST TERMINÉ AVEC SUCCÈS!")
    print(f"   Dataset: {len(df)} tirages historiques")
    print(f"   Prédiction: 1 combinaison générée")
    print(f"   Méthode: TimesFM avec wrapper loto spécialisé")

if __name__ == "__main__":
    test_système_complet()