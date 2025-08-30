"""
Wrapper TimesFM spécialisé pour les prédictions de loterie
Adapté spécifiquement pour les données de tirages de loto (pas commercial)
"""

import numpy as np
import sys
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import time

# Import du wrapper TimesFM original
sys.path.append(str(Path(__file__).parent.parent.parent / "timesfm_predict" / "models"))

try:
    from timesfm_wrapper import TimesFMPredictor as OriginalTimesFMPredictor
    TIMESFM_AVAILABLE = True
except ImportError as e:
    print(f"❌ Erreur d'import TimesFM: {e}")
    OriginalTimesFMPredictor = None
    TIMESFM_AVAILABLE = False


class LotoTimesFMPredictor:
    """
    Wrapper TimesFM spécialisé pour la prédiction de numéros de loterie
    Adaptations spécifiques pour les données de tirage (vs données commerciales)
    """
    
    def __init__(self, horizon_len: int = 1, backend: str = "cpu", model_repo: str = "google/timesfm-1.0-200m"):
        """
        Initialise le prédicteur TimesFM pour loterie
        
        Args:
            horizon_len: Nombre de tirages à prédire (généralement 1)
            backend: Backend de calcul ("cpu" ou "gpu")
            model_repo: Repository du modèle TimesFM à utiliser
        """
        self.horizon_len = horizon_len
        self.backend = backend
        self.model_repo = model_repo
        self.is_loaded = False
        self.original_predictor = None
        
        # Configuration spécifique au loto
        self.loto_config = {
            'enable_noise': True,  # Ajouter du bruit pour diversifier
            'noise_level': 0.05,   # Niveau de bruit (5%)
            'min_value': 1,        # Valeur minimale (boules: 1, chance: 1)
            'max_value': 49,       # Valeur maximale (boules: 49, chance: 10)
        }
    
    def load_model(self, simulation_mode: bool = False) -> bool:
        """Charge le modèle TimesFM"""
        if not TIMESFM_AVAILABLE and not simulation_mode:
            print("❌ TimesFM non disponible, basculement en mode simulation")
            simulation_mode = True
        
        if simulation_mode:
            print("🎲 Mode simulation loto activé")
            self.is_loaded = True
            return True
        
        try:
            # Créer le prédicteur TimesFM original
            self.original_predictor = OriginalTimesFMPredictor(
                horizon_len=self.horizon_len,
                backend=self.backend,
                model_repo=self.model_repo
            )
            
            # Charger le modèle original
            success = self.original_predictor.load_model(simulation_mode=False)
            # S'assurer que success est un booléen valide
            if success is None:
                success = True  # Si pas d'erreur et original_predictor existe
            
            self.is_loaded = bool(success)
            print(f"✅ Wrapper loto chargé: {self.is_loaded}")
            return self.is_loaded
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle loto: {e}")
            self.is_loaded = False
            return False
    
    def predict_loto_numbers(self, number_series: np.ndarray, 
                            component_type: str = "boule", 
                            auto_optimize: bool = True) -> Dict[str, Any]:
        """
        Prédit le prochain numéro pour une composante du loto
        
        Args:
            number_series: Série historique des numéros
            component_type: Type de composante ("boule" ou "chance") 
            auto_optimize: Activer les optimisations automatiques
            
        Returns:
            Dictionnaire avec prédictions et métadonnées
        """
        if not self.is_loaded:
            raise RuntimeError("Modèle non chargé. Appelez load_model() d'abord.")
        
        if self.original_predictor is None:
            # Mode simulation ou pas de TimesFM disponible
            return self._simulate_loto_prediction(number_series, component_type)
        
        # Préparation des données pour TimesFM
        optimized_series = self._optimize_loto_series(number_series, component_type)
        
        try:
            # Appel au TimesFM original (en tant que données de "ventes")
            result = self.original_predictor.predict_sales(
                optimized_series, 
                auto_optimize=auto_optimize
            )
            
            # Post-traitement spécialisé loto
            return self._post_process_loto_prediction(result, number_series, component_type)
            
        except Exception as e:
            print(f"❌ Erreur prédiction TimesFM loto: {e}")
            return self._simulate_loto_prediction(number_series, component_type)
    
    def _optimize_loto_series(self, series: np.ndarray, component_type: str) -> np.ndarray:
        """
        Optimise une série de numéros de loto pour TimesFM
        Pas d'optimisations commerciales, juste préparation pour l'IA
        """
        print("🔧 Optimisation série loto pour TimesFM")
        
        optimized = series.astype(float).copy()
        
        # 1. Ajout de variabilité contrôlée (pour diversifier les prédictions)
        if self.loto_config['enable_noise'] and len(optimized) > 10:
            noise_level = self.loto_config['noise_level']
            noise = np.random.normal(0, optimized.std() * noise_level, len(optimized))
            optimized += noise
            print(f"   🎲 Variabilité ajoutée: ±{noise_level*100:.1f}%")
        
        # 2. Normalisation douce (préserver les patterns)
        if len(optimized) > 5:
            # Lissage très léger pour réduire le bruit excessif
            smoothed = optimized.copy()
            for i in range(1, len(optimized) - 1):
                smoothed[i] = 0.8 * optimized[i] + 0.1 * optimized[i-1] + 0.1 * optimized[i+1]
            optimized = smoothed
            print("   📊 Lissage léger appliqué")
        
        # 3. Limitation contexte TimesFM (2048 points max)
        if len(optimized) > 2048:
            print(f"   ✂️  Série tronquée: {len(optimized)} → 2048 points")
            optimized = optimized[-2048:]
        
        print(f"   ✅ Série optimisée: {len(optimized)} points, μ={optimized.mean():.1f}, σ={optimized.std():.1f}")
        return optimized
    
    def _post_process_loto_prediction(self, timesfm_result: Dict, 
                                    original_series: np.ndarray,
                                    component_type: str) -> Dict[str, Any]:
        """Post-traitement des prédictions TimesFM pour le loto"""
        print("🎯 Post-traitement prédictions loto")
        
        # Extraire la prédiction brute
        if 'predictions' in timesfm_result and len(timesfm_result['predictions']) > 0:
            raw_prediction = timesfm_result['predictions'][0]
        else:
            # Fallback
            raw_prediction = original_series.mean()
        
        # Conversion en entier approprié pour le loto
        if component_type == "chance":
            # Numéro chance: 1-10
            processed = max(1, min(10, int(float(raw_prediction))))
        else:
            # Boules: 1-49
            processed = max(1, min(49, int(float(raw_prediction))))
        
        # Calcul de confiance basée sur la cohérence historique
        historical_mean = original_series.mean()
        historical_std = original_series.std()
        
        # Distance de la prédiction par rapport à l'historique
        distance = abs(float(raw_prediction) - historical_mean) / historical_std
        confidence = max(0.3, min(0.95, 1.0 - distance / 3.0))  # Entre 30% et 95%
        
        print(f"   📊 Brut: {float(raw_prediction):.2f} → Loto: {processed}")
        print(f"   🎯 Confiance: {confidence:.1%}")
        
        return {
            'predictions': [processed],
            'raw_prediction': raw_prediction,
            'confidence': confidence,
            'component_type': component_type,
            'historical_context': {
                'mean': float(historical_mean),
                'std': float(historical_std),
                'last_value': float(original_series[-1]),
                'trend': float(original_series[-5:].mean() - original_series[-10:-5].mean()) if len(original_series) >= 10 else 0.0
            },
            'processing_info': {
                'method': 'timesfm_loto',
                'model_repo': self.model_repo,
                'backend': self.backend
            }
        }
    
    def _simulate_loto_prediction(self, series: np.ndarray, component_type: str) -> Dict[str, Any]:
        """Génère une prédiction simulée pour le loto"""
        print("🎲 Simulation prédiction loto")
        
        if component_type == "chance":
            # Numéro chance: distribution plus uniforme
            prediction = np.random.randint(1, 11)
            confidence = 0.6
        else:
            # Boules: distribution basée sur l'historique
            mean_val = series.mean()
            std_val = series.std()
            
            # Prédiction avec bruit gaussien
            raw_pred = np.random.normal(mean_val, std_val * 0.3)
            prediction = max(1, min(49, int(round(raw_pred))))
            confidence = 0.7
        
        print(f"   🎯 Simulation: {prediction} (confiance: {confidence:.1%})")
        
        return {
            'predictions': [prediction],
            'raw_prediction': float(prediction),
            'confidence': confidence,
            'component_type': component_type,
            'historical_context': {
                'mean': float(series.mean()),
                'std': float(series.std()),
                'last_value': float(series[-1])
            },
            'processing_info': {
                'method': 'simulation_loto',
                'model_repo': 'simulation',
                'backend': 'cpu'
            }
        }
    
    def batch_predict_loto(self, series_dict: Dict[str, np.ndarray], 
                          num_variants: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Génère plusieurs variantes de prédictions pour diversifier
        
        Args:
            series_dict: Dictionnaire des séries par composante
            num_variants: Nombre de variantes à générer
            
        Returns:
            Dictionnaire des prédictions par composante
        """
        print(f"🎲 Génération de {num_variants} variantes par composante")
        
        results = {}
        
        for component, series in series_dict.items():
            component_type = "chance" if "chance" in component else "boule"
            variants = []
            
            for i in range(num_variants):
                print(f"   🔮 {component} - variante {i+1}")
                
                # Ajouter un peu de bruit différent à chaque variante
                original_noise = self.loto_config['enable_noise']
                self.loto_config['enable_noise'] = True
                
                try:
                    prediction = self.predict_loto_numbers(series, component_type, auto_optimize=True)
                    prediction['variant_id'] = i + 1
                    variants.append(prediction)
                except Exception as e:
                    print(f"   ❌ Erreur variante {i+1}: {e}")
                
                # Restaurer la configuration
                self.loto_config['enable_noise'] = original_noise
                
                # Petit délai pour diversifier les graines aléatoires
                time.sleep(0.01)
            
            results[component] = variants
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle"""
        return {
            'model_repo': self.model_repo,
            'backend': self.backend,
            'horizon_len': self.horizon_len,
            'is_loaded': self.is_loaded,
            'timesfm_available': TIMESFM_AVAILABLE,
            'mode': 'simulation' if self.original_predictor is None else 'timesfm_real',
            'loto_config': self.loto_config.copy()
        }