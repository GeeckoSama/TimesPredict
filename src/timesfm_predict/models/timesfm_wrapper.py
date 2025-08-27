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
                 backend: str = "gpu",
                 model_repo: str = "google/timesfm-2.0-500m-pytorch"):
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
        
    def load_model(self):
        """Charge le modèle TimesFM"""
        try:
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=self.backend,
                    horizon_len=self.horizon_len
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.model_repo
                )
            )
            print(f"Modèle TimesFM chargé avec succès (horizon: {self.horizon_len})")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            raise
            
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
            
        # Prépare les données d'entrée
        if sales_data.ndim == 1:
            forecast_input = [sales_data]
        else:
            forecast_input = [sales_data[i] for i in range(sales_data.shape[0])]
            
        # Fréquences par défaut
        if frequencies is None:
            frequencies = [0] * len(forecast_input)
            
        # Prédiction
        try:
            point_forecast = self.model.forecast(
                forecast_input, 
                freq=frequencies
            )
            
            result = {
                "predictions": point_forecast,
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
            
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle"""
        return {
            "horizon_len": self.horizon_len,
            "backend": self.backend,
            "model_repo": self.model_repo,
            "model_loaded": self.model is not None
        }