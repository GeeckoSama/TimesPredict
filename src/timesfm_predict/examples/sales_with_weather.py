"""
Exemple avancé : Prédiction de ventes avec données météorologiques
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from timesfm_predict.models.timesfm_wrapper import TimesFMPredictor
from timesfm_predict.data.sales_data import SalesDataProcessor
from timesfm_predict.data.weather_data import WeatherDataProcessor


def main():
    """Exemple de prédiction avec covariables météorologiques"""
    
    print("=== Exemple TimesFM : Ventes + Météo ===\n")
    
    # 1. Données de ventes
    print("1. Génération des données de ventes...")
    sales_processor = SalesDataProcessor()
    
    sales_data = sales_processor.create_sample_data(
        start_date="2023-01-01",
        periods=365,
        base_sales=2000.0,
        seasonality=True,
        trend=0.08,
        noise_level=0.12
    )
    
    print(f"Ventes générées: {len(sales_data)} jours\n")
    
    # 2. Données météorologiques
    print("2. Génération des données météorologiques...")
    weather_processor = WeatherDataProcessor()
    
    weather_data = weather_processor.generate_sample_weather_data(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        location_name="Boutique Centre-Ville"
    )
    
    print(f"Données météo générées: {len(weather_data)} jours\n")
    
    # 3. Combinaison des données
    print("3. Combinaison ventes + météo...")
    
    # Enrichissement des données météo
    enhanced_weather = weather_processor.create_weather_features()
    
    # Fusion avec les ventes
    combined_data = sales_data.merge(enhanced_weather, on='date', how='inner')
    
    print(f"Données combinées: {len(combined_data)} lignes avec {len(combined_data.columns)} colonnes")
    
    # Affiche quelques corrélations intéressantes
    correlations = combined_data[['sales', 'temperature', 'humidity', 'precipitation', 'wind_speed']].corr()['sales'].sort_values(ascending=False)
    print("\nCorrélations avec les ventes:")
    for var, corr in correlations.items():
        if var != 'sales':
            print(f"  {var}: {corr:.3f}")
    print()
    
    # 4. Préparation pour TimesFM
    print("4. Préparation des données...")
    
    # Prépare les données de ventes pour TimesFM
    context_length = 250
    sales_for_prediction = combined_data['sales'].values[-context_length:]
    
    # Prépare les covariables météo
    weather_covariates = weather_processor.prepare_covariates([
        'temperature', 'humidity', 'precipitation', 'wind_speed', 'comfort_index'
    ])
    
    # Ajuste la longueur des covariables
    for key in weather_covariates:
        weather_covariates[key] = weather_covariates[key][-context_length:]
    
    print(f"Contexte: {len(sales_for_prediction)} points")
    print(f"Covariables: {list(weather_covariates.keys())}")
    
    # 5. Modèle et prédiction
    print("\n5. Chargement du modèle TimesFM...")
    
    predictor = TimesFMPredictor(horizon_len=30, backend="cpu")
    
    try:
        predictor.load_model()
        
        print("6. Génération des prédictions...")
        
        # Prédiction simple (sans covariables pour l'instant)
        # Note: L'intégration complète des covariables dans TimesFM est expérimentale
        results = predictor.predict_sales(sales_for_prediction)
        predictions = results["predictions"][0]
        
        print(f"Prédictions: {len(predictions)} points")
        
    except Exception as e:
        print(f"Erreur avec le modèle: {e}")
        print("Simulation de prédictions pour la démonstration...")
        
        # Simulation basique pour la démonstration
        last_sales = sales_for_prediction[-30:].mean()
        trend = (sales_for_prediction[-10:].mean() - sales_for_prediction[-30:-20].mean()) / 20
        
        predictions = []
        for i in range(30):
            # Tendance + variation aléatoire
            pred = last_sales + (trend * i) + np.random.normal(0, last_sales * 0.1)
            predictions.append(max(pred, last_sales * 0.5))  # Minimum de sécurité
            
        predictions = np.array(predictions)
    
    # 6. Analyse de l'impact météo
    print("\n7. Analyse de l'impact météorologique...")
    
    # Génère des prédictions météo futures fictives pour l'exemple
    future_weather = weather_processor.generate_sample_weather_data(
        start_date=combined_data['date'].max() + timedelta(days=1),
        end_date=combined_data['date'].max() + timedelta(days=30)
    )
    
    print("Météo future simulée (exemple):")
    print(f"  Température moyenne: {future_weather['temperature'].mean():.1f}°C")
    print(f"  Précipitations moyennes: {future_weather['precipitation'].mean():.1f}mm")
    print(f"  Humidité moyenne: {future_weather['humidity'].mean():.0f}%")
    
    # 7. Visualisation avancée
    print("\n8. Visualisation des résultats...")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Graphique 1: Ventes et prédictions
    historical_context = combined_data.tail(60)
    future_dates = pd.date_range(
        start=combined_data['date'].max() + timedelta(days=1),
        periods=len(predictions)
    )
    
    axes[0].plot(historical_context['date'], historical_context['sales'], 
                 label='Ventes historiques', color='blue', linewidth=2)
    axes[0].plot(future_dates, predictions, 
                 label='Prédictions', color='red', linewidth=2, linestyle='--')
    axes[0].axvline(x=combined_data['date'].max(), color='green', linestyle=':', alpha=0.7)
    axes[0].set_title('Prédiction des Ventes avec Context Météorologique', fontsize=14)
    axes[0].set_ylabel('Ventes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Graphique 2: Température et ventes
    axes[1].plot(historical_context['date'], historical_context['temperature'], 
                 color='orange', label='Température', linewidth=2)
    ax1_twin = axes[1].twinx()
    ax1_twin.plot(historical_context['date'], historical_context['sales'], 
                  color='blue', alpha=0.6, label='Ventes')
    
    # Ajout de la météo future
    axes[1].plot(future_dates, future_weather['temperature'], 
                 color='orange', linestyle='--', alpha=0.7, label='Temp. future')
    
    axes[1].set_ylabel('Température (°C)', color='orange')
    ax1_twin.set_ylabel('Ventes', color='blue')
    axes[1].set_title('Relation Température - Ventes')
    axes[1].grid(True, alpha=0.3)
    
    # Graphique 3: Corrélation météo-ventes
    weather_vars = ['temperature', 'humidity', 'precipitation', 'wind_speed']
    corr_values = [correlations[var] for var in weather_vars]
    
    axes[2].bar(weather_vars, corr_values, 
                color=['red' if x > 0 else 'blue' for x in corr_values], alpha=0.7)
    axes[2].set_title('Impact des Variables Météorologiques sur les Ventes')
    axes[2].set_ylabel('Corrélation')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = "sales_prediction_with_weather.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {output_path}")
    
    plt.show()
    
    # 8. Rapport de synthèse
    print("\n=== RAPPORT DE SYNTHÈSE ===")
    print(f"Période analysée: {combined_data['date'].min()} à {combined_data['date'].max()}")
    print(f"Ventes moyennes historiques: {combined_data['sales'].mean():.2f}")
    print(f"Prédictions moyennes: {predictions.mean():.2f}")
    print(f"Variation attendue: {((predictions.mean() / combined_data['sales'].mean()) - 1) * 100:+.1f}%")
    
    print(f"\nImpact météorologique identifié:")
    print(f"- Variable la plus corrélée: {correlations.index[1]} ({correlations.iloc[1]:.3f})")
    print(f"- Température moyenne historique: {combined_data['temperature'].mean():.1f}°C")
    print(f"- Jours de pluie: {(combined_data['precipitation'] > 0).sum()} / {len(combined_data)}")
    
    # Recommandations basées sur la météo
    print(f"\nRecommandations basées sur la météo future:")
    hot_days = (future_weather['temperature'] > 25).sum()
    rainy_days = (future_weather['precipitation'] > 0).sum()
    
    if hot_days > 15:
        print("- Forte période de chaleur attendue → Augmentation possible des ventes")
    if rainy_days > 10:
        print("- Période pluvieuse attendue → Possible baisse des ventes")
    
    print("\n=== Fin de l'analyse ===")


if __name__ == "__main__":
    main()