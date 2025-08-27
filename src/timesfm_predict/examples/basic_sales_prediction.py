"""
Exemple de base : Prédiction de ventes avec TimesFM
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from timesfm_predict.models.timesfm_wrapper import TimesFMPredictor
from timesfm_predict.data.sales_data import SalesDataProcessor
from timesfm_predict.data.weather_data import WeatherDataProcessor


def main():
    """Exemple de prédiction de ventes de base"""
    
    print("=== Exemple TimesFM : Prédiction de ventes ===\n")
    
    # 1. Création des données d'exemple
    print("1. Génération des données de ventes d'exemple...")
    sales_processor = SalesDataProcessor()
    
    # Génère 1 an de données de ventes
    sales_data = sales_processor.create_sample_data(
        start_date="2023-01-01",
        periods=365,
        base_sales=1500.0,
        seasonality=True,
        trend=0.05,
        noise_level=0.15
    )
    
    print(f"Données créées: {len(sales_data)} jours")
    print(f"Ventes moyennes: {sales_data['sales'].mean():.2f}")
    print(f"Période: {sales_data['date'].min()} à {sales_data['date'].max()}\n")
    
    # 2. Préparation pour TimesFM
    print("2. Préparation des données pour TimesFM...")
    
    # Utilise les 300 derniers jours comme contexte
    sales_array, metadata = sales_processor.prepare_for_timesfm(context_length=300)
    
    print(f"Contexte: {len(sales_array)} points")
    print(f"Statistiques: moyenne={metadata['statistics']['mean']:.2f}, "
          f"écart-type={metadata['statistics']['std']:.2f}\n")
    
    # 3. Initialisation du modèle
    print("3. Initialisation du modèle TimesFM...")
    
    predictor = TimesFMPredictor(
        horizon_len=30,  # Prédiction sur 30 jours
        backend="cpu"    # Utilise CPU (changez en "gpu" si disponible)
    )
    
    try:
        print("Chargement du modèle (cela peut prendre quelques minutes)...")
        predictor.load_model()
        
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        print("Vérifiez que timesfm est bien installé avec: pip install timesfm[torch]")
        return
    
    # 4. Prédiction
    print("\n4. Génération des prédictions...")
    
    try:
        results = predictor.predict_sales(sales_array)
        predictions = results["predictions"][0]  # Premier (et seul) élément
        
        print(f"Prédictions générées: {len(predictions)} points")
        print(f"Prédictions moyennes: {predictions.mean():.2f}")
        print(f"Horizon de prédiction: {results['horizon_len']} jours\n")
        
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return
    
    # 5. Visualisation
    print("5. Visualisation des résultats...")
    
    # Prépare les dates
    last_date = sales_data['date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(len(predictions))]
    
    # Derniers 60 jours de données historiques pour le contexte
    historical_context = sales_data.tail(60)
    
    # Graphique
    plt.figure(figsize=(15, 8))
    
    # Données historiques
    plt.plot(historical_context['date'], historical_context['sales'], 
             label='Ventes historiques', color='blue', linewidth=2)
    
    # Prédictions
    plt.plot(future_dates, predictions, 
             label='Prédictions TimesFM', color='red', linewidth=2, linestyle='--')
    
    # Point de séparation
    plt.axvline(x=last_date, color='green', linestyle=':', alpha=0.7, 
                label='Début des prédictions')
    
    plt.title('Prédiction des Ventes avec TimesFM', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Ventes', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Sauvegarde
    output_path = "sales_prediction_basic.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {output_path}")
    
    plt.show()
    
    # 6. Résumé des résultats
    print("\n=== Résumé des résultats ===")
    print(f"Période historique: {metadata['date_range'][0]} à {metadata['date_range'][1]}")
    print(f"Ventes historiques moyennes: {metadata['statistics']['mean']:.2f}")
    print(f"Prédictions moyennes: {predictions.mean():.2f}")
    print(f"Variation prédite: {((predictions.mean() / metadata['statistics']['mean']) - 1) * 100:+.1f}%")
    
    # Statistiques des prédictions
    print(f"\nDétails des prédictions:")
    print(f"- Minimum prédit: {predictions.min():.2f}")
    print(f"- Maximum prédit: {predictions.max():.2f}")
    print(f"- Écart-type prédit: {predictions.std():.2f}")


if __name__ == "__main__":
    main()