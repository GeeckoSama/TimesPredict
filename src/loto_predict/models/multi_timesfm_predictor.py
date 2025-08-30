"""
Multi-TimesFM Predictor pour Loto
Utilise 6 modèles TimesFM coordonnés pour prédire les 5 boules + numéro chance
"""

import numpy as np
import sys
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Ajouter le chemin vers timesfm_wrapper  
sys.path.append(str(Path(__file__).parent.parent.parent / "timesfm_predict" / "models"))

try:
    # Import du wrapper loto depuis le même dossier
    from .loto_timesfm_wrapper import LotoTimesFMPredictor, TIMESFM_AVAILABLE
    print("✅ Wrapper TimesFM Loto chargé")
except ImportError as e:
    print(f"❌ Erreur d'import wrapper Loto TimesFM: {e}")
    print("💡 Fallback vers l'ancien système")
    LotoTimesFMPredictor = None
    TIMESFM_AVAILABLE = False


class MultiTimesFMPredictor:
    """
    Prédicteur multi-modèles TimesFM pour le loto français
    Utilise 6 modèles TimesFM séparés pour chaque composant du tirage
    """
    
    def __init__(self, 
                 model_repo: str = "google/timesfm-2.0-500m-pytorch",
                 backend: str = "cpu",
                 horizon_len: int = 1):
        """
        Initialise le prédicteur multi-TimesFM
        
        Args:
            model_repo: Repository TimesFM à utiliser
            backend: "cpu" ou "gpu"
            horizon_len: Nombre de tirages à prédire (généralement 1)
        """
        self.model_repo = model_repo
        self.backend = backend
        self.horizon_len = horizon_len
        self.predictors = {}
        self.is_loaded = False
        
        # Définition des composants à prédire
        self.components = [
            'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'numero_chance'
        ]
        
        # Configuration spécifique par composant
        self.component_configs = {
            'boule_1': {'min_val': 1, 'max_val': 49, 'type': 'boule'},
            'boule_2': {'min_val': 1, 'max_val': 49, 'type': 'boule'},
            'boule_3': {'min_val': 1, 'max_val': 49, 'type': 'boule'},
            'boule_4': {'min_val': 1, 'max_val': 49, 'type': 'boule'},
            'boule_5': {'min_val': 1, 'max_val': 49, 'type': 'boule'},
            'numero_chance': {'min_val': 1, 'max_val': 10, 'type': 'chance'}
        }
        
    def _get_max_context_length(self) -> int:
        """Détermine la taille maximale de contexte selon le modèle TimesFM utilisé"""
        if "2.0" in self.model_repo and "500m" in self.model_repo:
            return 2048  # TimesFM 2.0-500M supporte jusqu'à 2048 points
        elif "1.0" in self.model_repo and "200m" in self.model_repo:
            return 512   # TimesFM 1.0-200M supporte jusqu'à 512 points
        else:
            # Fallback conservateur pour modèles inconnus
            return 512
    
    def get_context_length_from_percentage(self, percentage: int, total_data_points: int) -> int:
        """
        Calcule la taille de contexte basée sur un pourcentage des données totales,
        limitée par la capacité maximale du modèle
        
        Args:
            percentage: Pourcentage des données à utiliser (10, 25, 50, 100)
            total_data_points: Nombre total de points dans le dataset
            
        Returns:
            Nombre de points de contexte à utiliser
        """
        max_context = self._get_max_context_length()
        
        # Calculer le contexte demandé selon le pourcentage
        if percentage == 100:
            requested_context = min(total_data_points, max_context)
        else:
            requested_context = min(
                int(total_data_points * percentage / 100),
                max_context
            )
        
        # S'assurer qu'on a au moins 10 points de contexte
        return max(10, requested_context)
    
    def load_models(self, simulation_mode: bool = False) -> bool:
        """Charge les 6 modèles TimesFM"""
        if not TIMESFM_AVAILABLE and not simulation_mode:
            print("❌ TimesFM n'est pas disponible. Basculement en mode simulation.")
            simulation_mode = True
            
        print(f"🚀 Chargement de {len(self.components)} modèles TimesFM...")
        print(f"   Repository: {self.model_repo}")
        print(f"   Backend: {self.backend}")
        print(f"   Mode: {'Simulation' if simulation_mode else 'TimesFM Réel'}")
        
        success_count = 0
        
        for component in self.components:
            try:
                print(f"   📊 Chargement modèle: {component}")
                
                # Créer un prédicteur TimesFM Loto pour ce composant
                if simulation_mode or LotoTimesFMPredictor is None:
                    # Mode simulation : créer un prédicteur mock
                    predictor = self._create_mock_predictor(component)
                else:
                    predictor = LotoTimesFMPredictor(
                        horizon_len=self.horizon_len,
                        backend=self.backend,
                        model_repo=self.model_repo
                    )
                
                # Charger le modèle
                if simulation_mode or LotoTimesFMPredictor is None:
                    predictor.load_model(simulation_mode=True)
                else:
                    predictor.load_model(simulation_mode=simulation_mode)
                self.predictors[component] = predictor
                success_count += 1
                
            except Exception as e:
                print(f"   ❌ Erreur chargement {component}: {e}")
                # En cas d'échec, utiliser le mode simulation pour ce composant
                try:
                    predictor = self._create_mock_predictor(component)
                    self.predictors[component] = predictor
                    print(f"   🔄 {component} basculé en mode simulation")
                except:
                    print(f"   💥 Échec total pour {component}")
        
        self.is_loaded = success_count > 0
        
        if self.is_loaded:
            print(f"✅ {success_count}/{len(self.components)} modèles chargés avec succès")
        else:
            print("❌ Aucun modèle n'a pu être chargé")
            
        return self.is_loaded
    
    def predict_next_combination(self, 
                                   time_series_data: Dict[str, np.ndarray], 
                                   context_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Prédit la prochaine combinaison loto
        
        Args:
            time_series_data: Dictionnaire avec les séries temporelles de chaque composant
            context_length: Nombre de points de contexte à utiliser (None = utilise tout)
            
        Returns:
            Dictionnaire avec les prédictions et métadonnées
        """
        if not self.is_loaded:
            raise ValueError("Modèles non chargés. Appelez load_models() d'abord.")
            
        print("🎯 Génération des prédictions multi-TimesFM...")
        
        # Déterminer les limites de contexte selon le modèle
        max_context_length = self._get_max_context_length()
        
        if context_length:
            print(f"📊 Contexte demandé: {context_length} points")
            if context_length > max_context_length:
                print(f"⚠️  Contexte réduit de {context_length} à {max_context_length} (limite modèle)")
                context_length = max_context_length
        else:
            print(f"📊 Contexte auto: utilisation de la taille maximale ({max_context_length})")
            context_length = max_context_length
        
        raw_predictions = {}
        processed_predictions = {}
        prediction_stats = {}
        
        # Prédire chaque composant individuellement
        for component in self.components:
            if component not in time_series_data:
                print(f"⚠️  Données manquantes pour {component}, utilisation de données synthétiques")
                # Générer des données synthétiques basiques
                time_series_data[component] = self._generate_synthetic_data(component)
            
            try:
                print(f"   🔮 Prédiction: {component}")
                
                # Obtenir les données pour ce composant avec contexte limité
                full_series = time_series_data[component]
                if len(full_series) > context_length:
                    series_data = full_series[-context_length:]  # Prendre les N derniers points
                    print(f"     📈 Contexte: {len(series_data)}/{len(full_series)} points (derniers)")
                else:
                    series_data = full_series
                    print(f"     📈 Contexte: {len(series_data)} points (complet)")
                
                # Déterminer le type de composant pour le wrapper loto
                component_type = "chance" if "chance" in component else "boule"
                
                # Prédire avec le modèle TimesFM Loto spécialisé
                if hasattr(self.predictors[component], 'predict_loto_numbers'):
                    result = self.predictors[component].predict_loto_numbers(
                        series_data, 
                        component_type=component_type,
                        auto_optimize=True
                    )
                else:
                    # Fallback pour les predictors mock
                    result = self.predictors[component].predict_sales(
                        series_data, 
                        auto_optimize=True
                    )
                # Extraire la prédiction brute
                raw_pred = result['predictions'][0]
                if hasattr(raw_pred, 'flatten'):
                    raw_pred = raw_pred.flatten()[0]  # Premier élément pour horizon=1
                elif isinstance(raw_pred, (list, tuple, np.ndarray)):
                    raw_pred = float(raw_pred[0])
                else:
                    raw_pred = float(raw_pred)
                
                raw_predictions[component] = raw_pred
                
                # Post-traitement spécifique au composant
                processed_pred = self._postprocess_prediction(
                    raw_pred, component, series_data
                )
                processed_predictions[component] = processed_pred
                
                # Statistiques de prédiction
                prediction_stats[component] = {
                    'raw_value': raw_pred,
                    'processed_value': processed_pred,
                    'series_mean': float(series_data.mean()),
                    'series_std': float(series_data.std()),
                    'last_value': float(series_data[-1]),
                    'confidence': self._calculate_confidence(raw_pred, series_data)
                }
                
                print(f"     Brut: {raw_pred:.2f} → Traité: {processed_pred}")
                
            except Exception as e:
                print(f"   ❌ Erreur prédiction {component}: {e}")
                # Fallback: utiliser la moyenne historique avec bruit
                fallback = self._fallback_prediction(component, time_series_data.get(component))
                processed_predictions[component] = fallback
                prediction_stats[component] = {
                    'raw_value': fallback,
                    'processed_value': fallback,
                    'error': str(e),
                    'method': 'fallback'
                }
        
        # Post-traitement global pour éviter les doublons dans les boules
        final_combination = self._resolve_duplicates(processed_predictions)
        
        # Génération du résultat final
        result = {
            'combination': {
                'boules': sorted([final_combination[f'boule_{i}'] for i in range(1, 6)]),
                'numero_chance': final_combination['numero_chance']
            },
            'raw_predictions': raw_predictions,
            'processed_predictions': processed_predictions,
            'final_combination': final_combination,
            'prediction_stats': prediction_stats,
            'model_info': {
                'repository': self.model_repo,
                'backend': self.backend,
                'horizon': self.horizon_len,
                'components_loaded': len(self.predictors)
            },
            'metadata': self._generate_metadata(final_combination, prediction_stats)
        }
        
        print(f"🎯 Prédiction terminée:")
        print(f"   Boules: {result['combination']['boules']}")
        print(f"   Chance: {result['combination']['numero_chance']}")
        
        return result
    
    def _postprocess_prediction(self, raw_pred: float, component: str, series_data: np.ndarray) -> int:
        """Post-traite une prédiction brute pour un composant"""
        config = self.component_configs[component]
        
        # 1. Arrondir à l'entier le plus proche
        rounded = round(raw_pred)
        
        # 2. Appliquer les contraintes min/max
        clamped = max(config['min_val'], min(config['max_val'], rounded))
        
        # 3. Ajustement intelligent basé sur l'historique
        if component.startswith('boule'):
            # Pour les boules, éviter les valeurs trop extrêmes si possible
            historical_mean = series_data.mean()
            historical_std = series_data.std()
            
            # Si la prédiction est très éloignée de la moyenne (> 2σ)
            if abs(clamped - historical_mean) > 2 * historical_std:
                # La ramener vers une zone plus probable
                if clamped > historical_mean:
                    clamped = min(clamped, int(historical_mean + 1.5 * historical_std))
                else:
                    clamped = max(clamped, int(historical_mean - 1.5 * historical_std))
        
        # 4. Contraintes finales
        return max(config['min_val'], min(config['max_val'], int(clamped)))
    
    def _resolve_duplicates(self, predictions: Dict[str, int]) -> Dict[str, int]:
        """Résout les doublons dans les prédictions de boules"""
        result = predictions.copy()
        
        # Extraire les boules et vérifier les doublons
        boules = [predictions[f'boule_{i}'] for i in range(1, 6)]
        
        # Tant qu'il y a des doublons
        attempts = 0
        while len(set(boules)) != len(boules) and attempts < 10:
            print(f"   🔄 Résolution doublons (tentative {attempts + 1})")
            
            # Identifier les doublons
            seen = set()
            duplicates = []
            
            for i, boule in enumerate(boules):
                if boule in seen:
                    duplicates.append(i)
                else:
                    seen.add(boule)
            
            # Remplacer les doublons par des valeurs proches mais différentes
            for duplicate_idx in duplicates:
                original_value = boules[duplicate_idx]
                
                # Essayer des valeurs proches
                for offset in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
                    new_value = original_value + offset
                    if 1 <= new_value <= 49 and new_value not in boules:
                        boules[duplicate_idx] = new_value
                        result[f'boule_{duplicate_idx + 1}'] = new_value
                        print(f"     Boule {duplicate_idx + 1}: {original_value} → {new_value}")
                        break
                else:
                    # Si aucune valeur proche ne marche, chercher n'importe où
                    available = set(range(1, 50)) - set(boules)
                    if available:
                        new_value = min(available)  # Prendre la plus petite disponible
                        boules[duplicate_idx] = new_value
                        result[f'boule_{duplicate_idx + 1}'] = new_value
                        print(f"     Boule {duplicate_idx + 1}: {original_value} → {new_value} (forcé)")
            
            attempts += 1
        
        if len(set(boules)) != len(boules):
            print("   ⚠️  Doublons persistants après 10 tentatives")
        
        return result
    
    def _calculate_confidence(self, prediction: float, series_data: np.ndarray) -> float:
        """Calcule une mesure de confiance pour la prédiction"""
        try:
            # Basé sur la distance à la moyenne historique
            mean_dist = abs(prediction - series_data.mean()) / series_data.std()
            
            # Plus c'est proche de la moyenne, plus la confiance est élevée
            confidence = max(0.0, 1.0 - mean_dist / 3.0)  # 3σ rule
            return min(1.0, confidence)
        except:
            return 0.5  # Confiance neutre en cas d'erreur
    
    def _generate_synthetic_data(self, component: str) -> np.ndarray:
        """Génère des données synthétiques pour un composant manquant"""
        config = self.component_configs[component]
        
        # Générer 100 points de données aléatoires dans la plage valide
        size = 100
        if component == 'numero_chance':
            # Distribution plus uniforme pour le numéro chance
            data = np.random.randint(config['min_val'], config['max_val'] + 1, size=size)
        else:
            # Distribution normale centrée pour les boules
            mean = (config['min_val'] + config['max_val']) / 2
            std = (config['max_val'] - config['min_val']) / 6  # ~99% dans la plage
            data = np.random.normal(mean, std, size=size)
            data = np.clip(data, config['min_val'], config['max_val']).astype(int)
        
        return data.astype(float)
    
    def _fallback_prediction(self, component: str, series_data: Optional[np.ndarray]) -> int:
        """Prédiction de fallback en cas d'erreur"""
        config = self.component_configs[component]
        
        if series_data is not None and len(series_data) > 0:
            # Utiliser la moyenne historique avec un peu de bruit
            mean_val = series_data.mean()
            noise = np.random.normal(0, series_data.std() * 0.1)
            prediction = mean_val + noise
        else:
            # Valeur aléatoire dans la plage
            prediction = np.random.randint(config['min_val'], config['max_val'] + 1)
        
        # Post-traitement
        return max(config['min_val'], min(config['max_val'], int(round(prediction))))
    
    def _generate_metadata(self, combination: Dict[str, int], stats: Dict[str, Any]) -> Dict[str, Any]:
        """Génère des métadonnées sur la prédiction"""
        boules = [combination[f'boule_{i}'] for i in range(1, 6)]
        
        return {
            'somme_boules': sum(boules),
            'moyenne_boules': np.mean(boules),
            'ecart_min_max': max(boules) - min(boules),
            'nb_pairs': sum(1 for b in boules if b % 2 == 0),
            'nb_impairs': sum(1 for b in boules if b % 2 == 1),
            'confiance_moyenne': np.mean([stats[comp].get('confidence', 0.5) for comp in self.components]),
            'methodes_utilisees': [stats[comp].get('method', 'timesfm') for comp in self.components]
        }
    
    def get_models_status(self) -> Dict[str, Any]:
        """Retourne le statut des modèles chargés"""
        return {
            'loaded': self.is_loaded,
            'total_models': len(self.components),
            'loaded_models': len(self.predictors),
            'components': {
                comp: comp in self.predictors for comp in self.components
            },
            'model_repo': self.model_repo,
            'backend': self.backend
        }
    
    def _create_mock_predictor(self, component: str):
        """Crée un prédicteur mock pour le mode simulation"""
        class MockPredictor:
            def __init__(self, component_name):
                self.component = component_name
                self.is_loaded = False
                
            def load_model(self, simulation_mode=False):
                self.is_loaded = True
                return True
                
            def predict_sales(self, data, auto_optimize=True):
                # Générer une prédiction aléatoire plausible
                config = self.parent.component_configs[self.component]
                if 'chance' in self.component:
                    prediction = np.random.randint(config['min_val'], config['max_val'] + 1)
                else:
                    # Pour les boules, utiliser une distribution plus réaliste
                    mean_val = (config['min_val'] + config['max_val']) / 2
                    std_val = (config['max_val'] - config['min_val']) / 6
                    prediction = np.random.normal(mean_val, std_val)
                    prediction = np.clip(prediction, config['min_val'], config['max_val'])
                
                return {
                    'predictions': [float(prediction)],
                    'simulation': True
                }
        
        mock = MockPredictor(component)
        mock.parent = self  # Référence vers le parent pour accéder aux configs
        return mock
    
    def batch_predict(self, time_series_data: Dict[str, np.ndarray], 
                     num_predictions: int = 5) -> List[Dict[str, Any]]:
        """Génère plusieurs prédictions pour diversité"""
        print(f"🎲 Génération de {num_predictions} prédictions diverses...")
        
        predictions = []
        for i in range(num_predictions):
            print(f"   Prédiction {i + 1}/{num_predictions}")
            try:
                # Ajouter un peu de bruit aux données pour diversifier
                noisy_data = {}
                for key, series in time_series_data.items():
                    noise_level = series.std() * 0.05  # 5% de bruit
                    noise = np.random.normal(0, noise_level, len(series))
                    noisy_data[key] = series + noise
                
                pred = self.predict_next_combination(noisy_data)
                pred['batch_id'] = i + 1
                predictions.append(pred)
                
            except Exception as e:
                print(f"     ❌ Erreur prédiction {i + 1}: {e}")
        
        print(f"✅ {len(predictions)} prédictions générées")
        return predictions