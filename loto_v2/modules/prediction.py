"""
Prediction Module - Prédictions pondérées Loto V2
Combine TimesFM fine-tuné avec pondération par fréquences historiques
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from .stats import LotoStatsAnalyzer
from .finetuning import LotoFineTuner
from .progress import loading_animation, UnifiedProgressBar


class LotoPredictor:
    """Générateur de prédictions loto pondérées"""
    
    def __init__(self, data_file: str = "../data/raw/loto_complet_fusionne.csv"):
        self.stats_analyzer = LotoStatsAnalyzer(data_file)
        self.fine_tuner = LotoFineTuner(data_file)
        self.probability_weights = None
        self.model_loaded = False
        self.weights_loaded = False
        
    def load_weights(self, silent: bool = False) -> bool:
        """Charge les poids de probabilité historiques"""
        if self.weights_loaded and self.probability_weights:
            return True
            
        weights = self.stats_analyzer.get_probability_weights()
        if weights and weights["boules"]:
            self.probability_weights = weights
            self.weights_loaded = True
            if not silent:
                print("✅ Poids probabilistes chargés")
            return True
        if not silent:
            print("❌ Aucun poids disponible - calculer stats d'abord")
        return False
    
    def predict_single_combination(self, silent: bool = False) -> Dict[str, any]:
        """Génère une prédiction simple pondérée"""
        if self.probability_weights is None:
            if not self.load_weights(silent=silent):
                # Fallback sans pondération
                return self._generate_random_combination()
        
        try:
            # Prédiction des 5 boules avec pondération
            boules = self._predict_weighted_boules()
            
            # Prédiction du numéro chance avec pondération
            chance = self._predict_weighted_chance()
            
            # Calcul de la confiance
            confidence = self._calculate_confidence(boules, chance)
            
            return {
                "boules": [int(b) for b in sorted(boules)],
                "chance": int(chance),
                "confidence": confidence,
                "method": "weighted_prediction"
            }
            
        except Exception as e:
            print(f"❌ Erreur prédiction: {e}")
            return self._generate_random_combination()
    
    def _predict_weighted_boules(self) -> List[int]:
        """Prédit 5 boules avec pondération probabiliste"""
        if not self.probability_weights:
            return random.sample(range(1, 50), 5)
        
        boules_probs = self.probability_weights["boules"]
        
        # Créer distribution pondérée
        numbers = list(boules_probs.keys())
        # S'assurer que numbers est une liste d'entiers
        numbers = [int(n) for n in numbers if isinstance(n, (int, str, np.integer)) and 1 <= int(n) <= 49]
        weights = [boules_probs.get(num, 0.02) for num in numbers]  # 0.02 = probabilité uniforme
        
        # Échantillonnage pondéré sans remise
        selected_boules = []
        available_numbers = numbers.copy()
        available_weights = weights.copy()
        
        for _ in range(5):
            if not available_numbers:
                break
                
            # Normaliser poids
            total_weight = sum(available_weights)
            normalized_weights = [w/total_weight for w in available_weights]
            
            # Sélection pondérée
            choice = np.random.choice(available_numbers, p=normalized_weights)
            selected_boules.append(choice)
            
            # Retirer de la liste
            idx = available_numbers.index(choice)
            available_numbers.pop(idx)
            available_weights.pop(idx)
        
        # Compléter si nécessaire
        while len(selected_boules) < 5:
            missing = [n for n in range(1, 50) if n not in selected_boules]
            selected_boules.append(random.choice(missing))
        
        return selected_boules[:5]
    
    def _predict_weighted_chance(self) -> int:
        """Prédit numéro chance avec pondération"""
        if not self.probability_weights:
            return random.randint(1, 10)
        
        chance_probs = self.probability_weights["chance"]
        
        if not chance_probs:
            return random.randint(1, 10)
        
        numbers = list(range(1, 11))
        weights = [chance_probs.get(num, 0.1) for num in numbers]  # 0.1 = probabilité uniforme
        
        # Normalisation
        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]
        
        return np.random.choice(numbers, p=normalized_weights)
    
    def _calculate_confidence(self, boules: List[int], chance: int) -> float:
        """Calcule la confiance de la prédiction"""
        if not self.probability_weights:
            return 0.3  # Confiance faible sans pondération
        
        # Confiance basée sur les probabilités historiques
        boules_conf = []
        for boule in boules:
            prob = self.probability_weights["boules"].get(boule, 0.02)
            boules_conf.append(prob)
        
        chance_conf = self.probability_weights["chance"].get(chance, 0.1)
        
        # Confiance globale (moyenne pondérée)
        avg_boule_conf = sum(boules_conf) / len(boules_conf)
        total_conf = (avg_boule_conf * 0.8) + (chance_conf * 0.2)
        
        # Normalisation entre 0.3 et 0.9
        normalized_conf = 0.3 + (total_conf * 2.0)
        return min(0.9, max(0.3, normalized_conf))
    
    def _generate_random_combination(self) -> Dict[str, any]:
        """Génère une combinaison aléatoire (fallback)"""
        return {
            "boules": sorted(random.sample(range(1, 50), 5)),
            "chance": random.randint(1, 10),
            "confidence": 0.2,
            "method": "random_fallback"
        }
    
    def ensure_model_loaded(self, use_finetuned: bool = True, silent: bool = False) -> bool:
        """S'assure que le modèle est chargé (une seule fois)"""
        if self.model_loaded:
            return True
            
        try:
            if use_finetuned:
                success = self.fine_tuner.load_finetuned_model()
            else:
                success = self.fine_tuner.load_base_model()
                
            if success:
                self.model_loaded = True
                return True
        except Exception as e:
            if not silent:
                print(f"❌ Erreur chargement modèle: {e}")
        return False
    
    def predict_with_timesfm(self, use_finetuned: bool = True, silent: bool = False) -> Dict[str, any]:
        """Prédiction combinée TimesFM + pondération statistique"""
        try:
            # Charger modèle TimesFM (une seule fois)
            if not self.ensure_model_loaded(use_finetuned, silent=silent):
                if not silent:
                    print("⚠️  Fallback vers prédiction statistique pure")
                return self.predict_single_combination(silent=silent)
            
            # Charger poids (une seule fois)
            if not self.load_weights(silent=silent):
                if not silent:
                    print("⚠️  Prédiction TimesFM sans pondération")
                return self._predict_timesfm_only()
            
            # Prédiction hybride : TimesFM + pondération
            timesfm_pred = self._predict_timesfm_only()
            weighted_pred = self.predict_single_combination(silent=True)
            
            # Fusion des deux approches
            hybrid_boules = self._merge_predictions(
                timesfm_pred["boules"], 
                weighted_pred["boules"]
            )
            
            # Chance : préférer la prédiction pondérée
            final_chance = weighted_pred["chance"]
            
            # Confiance combinée
            confidence = (timesfm_pred["confidence"] + weighted_pred["confidence"]) / 2
            
            return {
                "boules": [int(b) for b in sorted(hybrid_boules)],
                "chance": int(final_chance),
                "confidence": min(0.95, confidence + 0.1),  # Bonus pour hybride
                "method": "hybrid_timesfm_weighted"
            }
            
        except Exception as e:
            print(f"❌ Erreur prédiction TimesFM: {e}")
            return self.predict_single_combination()
    
    def _predict_timesfm_only(self) -> Dict[str, any]:
        """Prédiction pure TimesFM (simulation)"""
        # Prédiction TimesFM réelle sur séquences historiques
        stats_data = self.stats_analyzer.load_stats()
        if stats_data and 'historical_sequences' in stats_data:
            # Utiliser vraies séquences historiques comme input
            sequences = stats_data['historical_sequences']
            boules = []
            
            for i in range(5):
                if i < len(sequences):
                    input_seq = sequences[f'boule_{i+1}'][-100:]  # Dernières 100 valeurs
                    prediction = self.fine_tuner.predict_sequence(input_seq)
                    boule = max(1, min(49, int(prediction)))
                    if boule not in boules:
                        boules.append(boule)
            
            # Compléter si doublons ou données manquantes
            while len(boules) < 5:
                candidate = random.randint(1, 49)
                if candidate not in boules:
                    boules.append(candidate)
            
            # Prédiction pour numéro chance
            if 'numero_chance' in sequences:
                chance_seq = sequences['numero_chance'][-50:]
                chance = max(1, min(10, int(self.fine_tuner.predict_sequence(chance_seq))))
            else:
                chance = random.randint(1, 10)
        else:
            # Fallback sans données historiques
            boules = random.sample(range(1, 50), 5)
            chance = random.randint(1, 10)
        
        return {
            "boules": [int(b) for b in sorted(boules[:5])],
            "chance": int(chance),
            "confidence": 0.7,
            "method": "timesfm_only"
        }
    
    def _merge_predictions(self, pred1: List[int], pred2: List[int]) -> List[int]:
        """Fusionne deux prédictions de boules"""
        # Prendre le meilleur des deux mondes
        merged = []
        
        # Prendre alternativement des éléments de chaque prédiction
        all_candidates = pred1 + pred2
        
        for num in all_candidates:
            if num not in merged and len(merged) < 5:
                merged.append(num)
        
        # Compléter si nécessaire
        while len(merged) < 5:
            candidate = random.randint(1, 49)
            if candidate not in merged:
                merged.append(candidate)
        
        return merged[:5]