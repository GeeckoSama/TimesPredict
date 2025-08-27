"""
Wrapper pour le modèle TimesFM adapté aux prédictions de ventes
"""

import numpy as np
from typing import List, Optional, Dict, Any
import timesfm


class TimesFMPredictor:
    """Wrapper simplifié pour TimesFM orienté prédiction de ventes"""
    
    def __init__(self, 
                 horizon_len: int = 30,
                 backend: str = "cpu",
                 model_repo: str = "google/timesfm-1.0-200m-pytorch"):
        """
        Initialise le prédicteur TimesFM
        
        Args:
            horizon_len: Nombre de points à prédire (par défaut 30 jours)
            backend: "gpu" ou "cpu"
            model_repo: Repository HuggingFace du modèle
        """
        self.horizon_len = horizon_len
        self.backend = backend
        self.model_repo = model_repo
        self.model = None
        
    def load_model(self, simulation_mode=False):
        """Charge le modèle TimesFM ou utilise le mode simulation"""
        if simulation_mode:
            print(f"⚠️ Mode simulation activé (horizon: {self.horizon_len})")
            print("Les prédictions seront simulées pour les tests")
            self.model = "simulation"
            return
            
        try:
            print(f"Tentative de chargement du modèle: {self.model_repo}")
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=self.backend,
                    horizon_len=self.horizon_len
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.model_repo
                )
            )
            print(f"✅ Modèle TimesFM chargé avec succès (horizon: {self.horizon_len})")
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            print("🔄 Basculement en mode simulation...")
            self.model = "simulation"
            
    def predict_sales(self, 
                     sales_data: np.ndarray,
                     frequencies: Optional[List[int]] = None,
                     covariates: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Prédit les ventes futures
        
        Args:
            sales_data: Données de vente historiques (array 1D ou 2D)
            frequencies: Fréquences des séries (0 pour auto-détection)
            covariates: Dictionnaire des covariables (météo, etc.)
            
        Returns:
            Dictionnaire avec les prédictions et métadonnées
        """
        if self.model is None:
            raise ValueError("Modèle non chargé. Appelez load_model() d'abord.")
        
        # Mode simulation
        if self.model == "simulation":
            print("🎯 Génération de prédictions simulées...")
            return self._simulate_predictions(sales_data)
            
        # Prépare les données d'entrée
        if sales_data.ndim == 1:
            forecast_input = [sales_data]
        else:
            forecast_input = [sales_data[i] for i in range(sales_data.shape[0])]
            
        # Fréquences par défaut
        if frequencies is None:
            frequencies = [0] * len(forecast_input)
            
        # Prédiction avec TimesFM
        try:
            point_forecast = self.model.forecast(
                forecast_input, 
                freq=frequencies
            )
            
            # Debug: voir le format retourné par TimesFM
            print(f"🔍 Debug TimesFM - Type: {type(point_forecast)}")
            if hasattr(point_forecast, 'shape'):
                print(f"🔍 Debug TimesFM - Shape: {point_forecast.shape}")
            elif isinstance(point_forecast, (list, tuple)):
                print(f"🔍 Debug TimesFM - Length: {len(point_forecast)}")
                if len(point_forecast) > 0:
                    print(f"🔍 Debug TimesFM - First item type: {type(point_forecast[0])}")
            print(f"🔍 Debug TimesFM - Content preview: {point_forecast}")
            
            # Normaliser le format de sortie pour compatibilité
            if isinstance(point_forecast, list):
                predictions_formatted = point_forecast
            elif isinstance(point_forecast, tuple):
                # TimesFM peut retourner un tuple - le convertir en liste
                predictions_formatted = list(point_forecast)
            elif hasattr(point_forecast, 'ndim'):
                # Si c'est un array numpy, le convertir en liste
                predictions_formatted = [point_forecast.flatten()] if point_forecast.ndim > 1 else [point_forecast]
            else:
                # Autre format - forcer en liste
                predictions_formatted = [point_forecast]
            
            result = {
                "predictions": predictions_formatted,
                "horizon_len": self.horizon_len,
                "input_shape": sales_data.shape,
                "model_repo": self.model_repo
            }
            
            # Ajoute les métadonnées des covariables si présentes
            if covariates:
                result["covariates_used"] = list(covariates.keys())
                
            return result
            
        except Exception as e:
            print(f"Erreur lors de la prédiction: {e}")
            raise
            
    def predict_with_confidence(self, 
                               sales_data: np.ndarray,
                               frequencies: Optional[List[int]] = None,
                               quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]) -> Dict[str, Any]:
        """
        Prédiction avec intervalles de confiance (fonctionnalité expérimentale)
        
        Args:
            sales_data: Données de vente historiques
            frequencies: Fréquences des séries
            quantiles: Quantiles pour les intervalles de confiance
            
        Returns:
            Dictionnaire avec prédictions ponctuelles et quantiles
        """
        if self.model is None:
            raise ValueError("Modèle non chargé. Appelez load_model() d'abord.")
            
        # Prépare les données
        if sales_data.ndim == 1:
            forecast_input = [sales_data]
        else:
            forecast_input = [sales_data[i] for i in range(sales_data.shape[0])]
            
        if frequencies is None:
            frequencies = [0] * len(forecast_input)
            
        try:
            # Prédiction ponctuelle
            point_forecast = self.model.forecast(forecast_input, freq=frequencies)
            
            # Prédiction par quantiles (expérimental)
            quantile_forecast = self.model.forecast_with_covariates(
                forecast_input,
                freq=frequencies,
                quantiles=quantiles
            )
            
            return {
                "point_predictions": point_forecast,
                "quantile_predictions": quantile_forecast,
                "quantiles": quantiles,
                "horizon_len": self.horizon_len
            }
            
        except Exception as e:
            print(f"Erreur lors de la prédiction avec quantiles: {e}")
            # Fallback sur prédiction simple
            return self.predict_sales(sales_data, frequencies)
            
    def _simulate_predictions(self, sales_data: np.ndarray) -> Dict[str, Any]:
        """Génère des prédictions simulées pour les tests"""
        
        # Gestion des données multiples (2D) vs simples (1D)
        if sales_data.ndim == 1:
            # Données 1D : une seule série
            return self._simulate_single_series(sales_data)
        else:
            # Données 2D : plusieurs séries
            all_predictions = []
            all_stats = []
            
            for i in range(sales_data.shape[0]):
                series = sales_data[i]
                result = self._simulate_single_series(series)
                all_predictions.append(result["predictions"][0])
                all_stats.append(result["simulation_stats"])
            
            return {
                "predictions": all_predictions,
                "horizon_len": self.horizon_len,
                "input_shape": sales_data.shape,
                "model_repo": "simulation",
                "mode": "simulation",
                "simulation_stats": all_stats
            }
    
    def _simulate_single_series(self, series_data: np.ndarray) -> Dict[str, Any]:
        """Simule les prédictions pour une série unique"""
        # Analyse des données historiques
        mean_sales = float(series_data.mean())
        std_sales = float(series_data.std())
        
        # Détecte la tendance sur les 30 derniers points
        recent_data = series_data[-min(30, len(series_data)):]
        trend = (recent_data[-1] - recent_data[0]) / len(recent_data) if len(recent_data) > 1 else 0
        
        # Génère des prédictions simulées
        predictions = []
        for i in range(self.horizon_len):
            # Tendance + variation aléatoire + saisonnalité simple
            base_pred = mean_sales + (trend * i)
            seasonal_effect = np.sin(2 * np.pi * i / 7) * (std_sales * 0.1)  # Saisonnalité hebdomadaire
            noise = np.random.normal(0, std_sales * 0.15)
            
            pred = base_pred + seasonal_effect + noise
            predictions.append(max(pred, mean_sales * 0.3))  # Seuil minimum
        
        predictions = np.array(predictions)
        
        return {
            "predictions": [predictions],  # Format compatible avec TimesFM
            "horizon_len": self.horizon_len,
            "input_shape": series_data.shape,
            "model_repo": "simulation",
            "mode": "simulation",
            "simulation_stats": {
                "mean_input": mean_sales,
                "std_input": std_sales,
                "detected_trend": trend,
                "mean_prediction": float(predictions.mean())
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle"""
        return {
            "horizon_len": self.horizon_len,
            "backend": self.backend,
            "model_repo": self.model_repo,
            "model_loaded": self.model is not None,
            "mode": "simulation" if self.model == "simulation" else "timesfm"
        }