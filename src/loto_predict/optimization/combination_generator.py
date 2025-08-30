"""
Générateur de combinaisons optimisé pour le loto français
Post-traite les prédictions TimesFM et optimise les combinaisons finales
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from itertools import combinations
import random
from collections import Counter


class CombinationGenerator:
    """
    Générateur intelligent de combinaisons loto
    Optimise les prédictions brutes selon différentes stratégies
    """
    
    def __init__(self, historical_data: Optional[pd.DataFrame] = None):
        """
        Initialise le générateur de combinaisons
        
        Args:
            historical_data: Données historiques pour l'optimisation
        """
        self.historical_data = historical_data
        self.strategies = {
            'timesfm_direct': self._strategy_timesfm_direct,
            'statistical_weighted': self._strategy_statistical_weighted,
            'hybrid_optimized': self._strategy_hybrid_optimized,
            'frequency_balanced': self._strategy_frequency_balanced,
            'pattern_aware': self._strategy_pattern_aware
        }
    
    def generate_combinations(self, 
                            timesfm_predictions: Dict[str, Any],
                            statistical_analysis: Dict[str, Any],
                            num_combinations: int = 5,
                            strategies: List[str] = None) -> Dict[str, Any]:
        """
        Génère des combinaisons optimisées selon différentes stratégies
        
        Args:
            timesfm_predictions: Résultats des prédictions TimesFM
            statistical_analysis: Analyse statistique des patterns
            num_combinations: Nombre de combinaisons à générer
            strategies: Liste des stratégies à utiliser
            
        Returns:
            Dict avec les combinaisons générées et leurs scores
        """
        if strategies is None:
            strategies = ['timesfm_direct', 'statistical_weighted', 'hybrid_optimized']
        
        print(f"🎯 Génération de {num_combinations} combinaisons avec {len(strategies)} stratégies...")
        
        all_combinations = []
        
        for strategy_name in strategies:
            if strategy_name in self.strategies:
                print(f"   📊 Stratégie: {strategy_name}")
                
                strategy_func = self.strategies[strategy_name]
                strategy_combinations = strategy_func(
                    timesfm_predictions, 
                    statistical_analysis, 
                    num_combinations
                )
                
                # Ajouter métadonnées
                for i, combo in enumerate(strategy_combinations):
                    combo['strategy'] = strategy_name
                    combo['strategy_rank'] = i + 1
                    combo['global_id'] = len(all_combinations) + 1
                    all_combinations.append(combo)
        
        # Éliminer les doublons exact
        unique_combinations = self._remove_exact_duplicates(all_combinations)
        
        # Classer par score global
        ranked_combinations = sorted(unique_combinations, 
                                   key=lambda x: x.get('score', 0), 
                                   reverse=True)
        
        # Limiter au nombre demandé
        final_combinations = ranked_combinations[:num_combinations]
        
        # Recalculer les rangs finaux
        for i, combo in enumerate(final_combinations):
            combo['final_rank'] = i + 1
        
        print(f"✅ {len(final_combinations)} combinaisons uniques générées")
        
        return {
            'combinations': final_combinations,
            'generation_stats': {
                'total_generated': len(all_combinations),
                'unique_combinations': len(unique_combinations),
                'final_selected': len(final_combinations),
                'strategies_used': strategies
            },
            'metadata': self._generate_metadata(final_combinations, statistical_analysis)
        }
    
    def _strategy_timesfm_direct(self, 
                                timesfm_pred: Dict[str, Any], 
                                stats: Dict[str, Any], 
                                num_combos: int) -> List[Dict[str, Any]]:
        """Stratégie directe basée sur les prédictions TimesFM"""
        
        combinations = []
        base_combination = timesfm_pred.get('final_combination', {})
        
        if not base_combination:
            return self._generate_fallback_combinations(num_combos)
        
        # Combinaison principale (prédiction directe)
        main_combo = {
            'boules': sorted([base_combination.get(f'boule_{i}', i) for i in range(1, 6)]),
            'numero_chance': base_combination.get('numero_chance', 1),
            'score': self._calculate_timesfm_score(timesfm_pred),
            'method': 'timesfm_direct_main',
            'confidence': timesfm_pred.get('metadata', {}).get('confiance_moyenne', 0.5)
        }
        combinations.append(main_combo)
        
        # Variations de la combinaison principale
        for i in range(1, min(num_combos, 4)):
            variant = self._create_timesfm_variant(base_combination, i, timesfm_pred)
            combinations.append(variant)
        
        # Compléter avec des combinaisons aléatoires pondérées si nécessaire
        while len(combinations) < num_combos:
            random_combo = self._generate_weighted_random_combo(timesfm_pred, len(combinations))
            combinations.append(random_combo)
        
        return combinations[:num_combos]
    
    def _generate_fallback_combinations(self, num_combos: int) -> List[Dict[str, Any]]:
        """Génère des combinaisons de fallback en cas d'erreur"""
        
        combinations = []
        
        for i in range(num_combos):
            boules = random.sample(range(1, 50), 5)
            chance = random.randint(1, 10)
            
            combo = {
                'boules': sorted(boules),
                'numero_chance': chance,
                'score': 0.25,  # Score faible pour les fallbacks
                'method': f'fallback_{i+1}',
                'confidence': 0.25
            }
            combinations.append(combo)
        
        return combinations

    def _calculate_timesfm_score(self, timesfm_pred: Dict[str, Any]) -> float:
        """Calcule le score d'une prédiction TimesFM"""
        base_score = 0.5
        
        if 'metadata' in timesfm_pred:
            metadata = timesfm_pred['metadata']
            
            # Bonus pour confiance élevée
            confidence = metadata.get('confiance_moyenne', 0.5)
            base_score += confidence * 0.3
            
            # Bonus pour somme dans la plage normale (100-200)
            somme = metadata.get('somme_boules', 125)
            if 100 <= somme <= 200:
                base_score += 0.2
            
            # Bonus pour parité équilibrée (2-3 pairs)
            nb_pairs = metadata.get('nb_pairs', 2.5)
            if 2 <= nb_pairs <= 3:
                base_score += 0.1
        
        return min(1.0, base_score)

    def _create_timesfm_variant(self, base_combination: Dict[str, int], 
                               variant_idx: int, timesfm_pred: Dict[str, Any]) -> Dict[str, Any]:
        """Crée une variante de la combinaison TimesFM de base"""
        
        variant_boules = [base_combination.get(f'boule_{i}', i) for i in range(1, 6)]
        
        # Modifier 1-2 boules selon l'index de variante
        num_changes = min(variant_idx, 2)
        
        for _ in range(num_changes):
            # Choisir une position à modifier
            pos_to_change = random.randint(0, 4)
            original_val = variant_boules[pos_to_change]
            
            # Générer une nouvelle valeur proche
            offset = random.choice([-3, -2, -1, 1, 2, 3])
            new_val = max(1, min(49, original_val + offset))
            
            # S'assurer qu'elle n'est pas déjà utilisée
            while new_val in variant_boules:
                new_val = random.randint(1, 49)
            
            variant_boules[pos_to_change] = new_val
        
        return {
            'boules': sorted(variant_boules),
            'numero_chance': base_combination.get('numero_chance', random.randint(1, 10)),
            'score': self._calculate_timesfm_score(timesfm_pred) * (1 - variant_idx * 0.1),
            'method': f'timesfm_variant_{variant_idx}',
            'confidence': timesfm_pred.get('metadata', {}).get('confiance_moyenne', 0.5) * 0.9
        }

    def _generate_weighted_random_combo(self, timesfm_pred: Dict[str, Any], combo_idx: int) -> Dict[str, Any]:
        """Génère une combinaison aléatoire pondérée"""
        
        boules = random.sample(range(1, 50), 5)
        chance = random.randint(1, 10)
        
        return {
            'boules': sorted(boules),
            'numero_chance': chance,
            'score': 0.3 + random.random() * 0.2,  # Score aléatoire modéré
            'method': f'weighted_random_{combo_idx}',
            'confidence': 0.4
        }

    def _strategy_statistical_weighted(self, 
                                     timesfm_pred: Dict[str, Any], 
                                     stats: Dict[str, Any], 
                                     num_combos: int) -> List[Dict[str, Any]]:
        """Stratégie basée sur les statistiques historiques"""
        
        combinations = []
        
        if 'frequencies' not in stats:
            return self._generate_fallback_combinations(num_combos)
        
        freq_data = stats['frequencies']
        boules_freq = freq_data.get('boules_frequencies', {})
        chance_freq = freq_data.get('chance_frequencies', {})
        
        # Créer des poids basés sur les fréquences inversées (favorise les moins sortis)
        boules_weights = self._calculate_frequency_weights(boules_freq, inverse=True)
        chance_weights = self._calculate_frequency_weights(chance_freq, inverse=True)
        
        for i in range(num_combos):
            # Sélectionner 5 boules selon les poids
            selected_boules = self._weighted_selection(boules_weights, 5)
            selected_chance = self._weighted_selection(chance_weights, 1)[0]
            
            combo = {
                'boules': sorted(selected_boules),
                'numero_chance': selected_chance,
                'score': self._calculate_statistical_score(selected_boules, selected_chance, stats),
                'method': f'statistical_weighted_{i+1}',
                'confidence': 0.6  # Confiance moyenne pour les stats
            }
            combinations.append(combo)
        
        return combinations

    def _strategy_hybrid_optimized(self, 
                                 timesfm_pred: Dict[str, Any], 
                                 stats: Dict[str, Any], 
                                 num_combos: int) -> List[Dict[str, Any]]:
        """Stratégie hybride combinant TimesFM et statistiques"""
        
        combinations = []
        base_combination = timesfm_pred.get('final_combination', {})
        
        if not base_combination or 'frequencies' not in stats:
            return self._generate_fallback_combinations(num_combos)
        
        freq_data = stats['frequencies']
        boules_freq = freq_data.get('boules_frequencies', {})
        
        for i in range(num_combos):
            # Mélanger prédictions TimesFM avec insights statistiques
            hybrid_boules = []
            
            # Commencer avec 2-3 boules de TimesFM
            timesfm_boules = [base_combination.get(f'boule_{j}', j) for j in range(1, 6)]
            selected_from_timesfm = random.sample(timesfm_boules, min(3, len(timesfm_boules)))
            hybrid_boules.extend(selected_from_timesfm)
            
            # Compléter avec des boules statistiquement optimisées
            available_boules = [b for b in range(1, 50) if b not in hybrid_boules]
            
            # Favoriser les numéros avec gap élevé (pas sortis récemment)
            if 'sequences' in stats and 'gaps_analysis' in stats['sequences']:
                gaps_data = stats['sequences']['gaps_analysis']
                available_boules.sort(key=lambda x: gaps_data.get(x, {}).get('since_last', 0), reverse=True)
            
            # Ajouter les boules manquantes
            needed = 5 - len(hybrid_boules)
            hybrid_boules.extend(available_boules[:needed])
            
            # Numéro chance hybride
            timesfm_chance = base_combination.get('numero_chance', 1)
            if random.random() < 0.7:  # 70% chance d'utiliser TimesFM
                hybrid_chance = timesfm_chance
            else:
                # Choisir un numéro chance différent
                chance_options = [c for c in range(1, 11) if c != timesfm_chance]
                hybrid_chance = random.choice(chance_options)
            
            combo = {
                'boules': sorted(hybrid_boules[:5]),
                'numero_chance': hybrid_chance,
                'score': self._calculate_hybrid_score(hybrid_boules[:5], hybrid_chance, timesfm_pred, stats),
                'method': f'hybrid_optimized_{i+1}',
                'confidence': 0.75  # Confiance élevée pour l'hybride
            }
            combinations.append(combo)
        
        return combinations

    def _strategy_frequency_balanced(self, 
                                   timesfm_pred: Dict[str, Any], 
                                   stats: Dict[str, Any], 
                                   num_combos: int) -> List[Dict[str, Any]]:
        """Stratégie équilibrée basée sur les fréquences"""
        
        combinations = []
        
        if 'frequencies' not in stats:
            return self._generate_fallback_combinations(num_combos)
        
        freq_data = stats['frequencies']
        hot_numbers = freq_data.get('hot_numbers', [])
        cold_numbers = freq_data.get('cold_numbers', [])
        
        # Numéros modérés (ni chauds ni froids)
        all_numbers = set(range(1, 50))
        moderate_numbers = list(all_numbers - set(hot_numbers) - set(cold_numbers))
        
        for i in range(num_combos):
            balanced_boules = []
            
            # Répartition équilibrée
            if hot_numbers:
                balanced_boules.append(random.choice(hot_numbers))  # 1 chaud
            if cold_numbers:
                balanced_boules.append(random.choice(cold_numbers))  # 1 froid
            
            # Compléter avec des modérés
            while len(balanced_boules) < 5 and moderate_numbers:
                choice = random.choice(moderate_numbers)
                if choice not in balanced_boules:
                    balanced_boules.append(choice)
                    moderate_numbers.remove(choice)
            
            # Si pas assez de modérés, ajouter n'importe quoi
            while len(balanced_boules) < 5:
                choice = random.randint(1, 49)
                if choice not in balanced_boules:
                    balanced_boules.append(choice)
            
            combo = {
                'boules': sorted(balanced_boules),
                'numero_chance': random.randint(1, 10),
                'score': self._calculate_balance_score(balanced_boules, hot_numbers, cold_numbers),
                'method': f'frequency_balanced_{i+1}',
                'confidence': 0.65
            }
            combinations.append(combo)
        
        return combinations

    def _strategy_pattern_aware(self, 
                              timesfm_pred: Dict[str, Any], 
                              stats: Dict[str, Any], 
                              num_combos: int) -> List[Dict[str, Any]]:
        """Stratégie consciente des patterns temporels et séquentiels"""
        
        combinations = []
        
        # Éviter les patterns trop courants
        for i in range(num_combos):
            pattern_boules = []
            
            # Éviter trop de consécutifs
            available = list(range(1, 50))
            
            for _ in range(5):
                if available:
                    # Choisir un numéro
                    choice = random.choice(available)
                    pattern_boules.append(choice)
                    
                    # Supprimer les numéros trop proches pour éviter les consécutifs
                    to_remove = [choice]
                    if len(pattern_boules) < 3:  # Permettre maximum 2 consécutifs
                        to_remove.extend([choice-1, choice+1])
                    
                    available = [x for x in available if x not in to_remove]
            
            # Ajustement de la parité (éviter 5 pairs ou 5 impairs)
            pairs_count = sum(1 for b in pattern_boules if b % 2 == 0)
            if pairs_count == 0 or pairs_count == 5:
                # Remplacer un numéro pour équilibrer
                if pairs_count == 0:
                    # Remplacer un impair par un pair
                    impair_idx = next(i for i, b in enumerate(pattern_boules) if b % 2 == 1)
                    new_pair = next(b for b in range(2, 50, 2) if b not in pattern_boules)
                    pattern_boules[impair_idx] = new_pair
                else:
                    # Remplacer un pair par un impair
                    pair_idx = next(i for i, b in enumerate(pattern_boules) if b % 2 == 0)
                    new_impair = next(b for b in range(1, 50, 2) if b not in pattern_boules)
                    pattern_boules[pair_idx] = new_impair
            
            combo = {
                'boules': sorted(pattern_boules),
                'numero_chance': random.randint(1, 10),
                'score': self._calculate_pattern_score(pattern_boules, stats),
                'method': f'pattern_aware_{i+1}',
                'confidence': 0.6
            }
            combinations.append(combo)
        
        return combinations

    def _calculate_statistical_score(self, boules: List[int], chance: int, stats: Dict[str, Any]) -> float:
        """Calcule le score basé sur les statistiques"""
        base_score = 0.5
        
        if 'frequencies' in stats:
            freq_data = stats['frequencies']
            boules_freq = freq_data.get('boules_frequencies', {})
            
            # Score basé sur la diversité des fréquences
            frequencies = [boules_freq.get(b, 0) for b in boules]
            if frequencies:
                # Favoriser une distribution variée des fréquences
                freq_std = np.std(frequencies)
                base_score += min(0.3, freq_std / 50)
        
        return min(1.0, base_score)

    def _calculate_hybrid_score(self, boules: List[int], chance: int, 
                              timesfm_pred: Dict[str, Any], stats: Dict[str, Any]) -> float:
        """Calcule le score pour une stratégie hybride"""
        timesfm_score = self._calculate_timesfm_score(timesfm_pred) * 0.6
        statistical_score = self._calculate_statistical_score(boules, chance, stats) * 0.4
        
        return timesfm_score + statistical_score

    def _calculate_balance_score(self, boules: List[int], hot_numbers: List[int], cold_numbers: List[int]) -> float:
        """Calcule le score d'équilibre fréquentiel"""
        base_score = 0.5
        
        hot_count = sum(1 for b in boules if b in hot_numbers)
        cold_count = sum(1 for b in boules if b in cold_numbers)
        moderate_count = 5 - hot_count - cold_count
        
        # Favoriser un bon équilibre
        if moderate_count >= 3:  # Majorité de modérés
            base_score += 0.3
        if hot_count == 1 and cold_count == 1:  # 1 chaud, 1 froid
            base_score += 0.2
        
        return min(1.0, base_score)

    def _calculate_pattern_score(self, boules: List[int], stats: Dict[str, Any]) -> float:
        """Calcule le score basé sur les patterns"""
        base_score = 0.5
        
        # Bonus pour éviter trop de consécutifs
        consecutive_count = 0
        sorted_boules = sorted(boules)
        for i in range(len(sorted_boules) - 1):
            if sorted_boules[i+1] == sorted_boules[i] + 1:
                consecutive_count += 1
        
        if consecutive_count <= 1:  # Maximum 1 paire consécutive
            base_score += 0.2
        
        # Bonus pour parité équilibrée
        pairs_count = sum(1 for b in boules if b % 2 == 0)
        if 2 <= pairs_count <= 3:
            base_score += 0.1
        
        return min(1.0, base_score)

    def _calculate_frequency_weights(self, freq_dict: Dict[int, int], inverse: bool = False) -> Dict[int, float]:
        """Calcule les poids basés sur les fréquences"""
        if not freq_dict:
            return {}
        
        max_freq = max(freq_dict.values())
        min_freq = min(freq_dict.values())
        
        weights = {}
        for num, freq in freq_dict.items():
            if inverse:
                # Plus la fréquence est faible, plus le poids est élevé
                weights[num] = (max_freq - freq + 1) / (max_freq - min_freq + 1)
            else:
                # Plus la fréquence est élevée, plus le poids est élevé
                weights[num] = freq / max_freq
        
        return weights

    def _weighted_selection(self, weights: Dict[int, float], count: int) -> List[int]:
        """Sélection pondérée sans remplacement"""
        if not weights:
            return random.sample(range(1, 50), min(count, 49))
        
        selected = []
        available_weights = weights.copy()
        
        for _ in range(count):
            if not available_weights:
                break
            
            # Sélection pondérée
            total_weight = sum(available_weights.values())
            rand_val = random.random() * total_weight
            
            cumsum = 0
            for num, weight in available_weights.items():
                cumsum += weight
                if rand_val <= cumsum:
                    selected.append(num)
                    del available_weights[num]
                    break
        
        return selected

    def _remove_exact_duplicates(self, combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Supprime les combinaisons exactement identiques"""
        
        seen_combinations = set()
        unique_combinations = []
        
        for combo in combinations:
            # Créer une signature unique
            signature = (tuple(combo['boules']), combo['numero_chance'])
            
            if signature not in seen_combinations:
                seen_combinations.add(signature)
                unique_combinations.append(combo)
        
        return unique_combinations

    def _generate_metadata(self, combinations: List[Dict[str, Any]], 
                         stats: Dict[str, Any]) -> Dict[str, Any]:
        """Génère des métadonnées sur les combinaisons générées"""
        
        if not combinations:
            return {}
        
        # Statistiques des combinaisons
        all_boules = []
        for combo in combinations:
            all_boules.extend(combo['boules'])
        
        boules_counter = Counter(all_boules)
        chances_counter = Counter([combo['numero_chance'] for combo in combinations])
        
        return {
            'total_combinations': len(combinations),
            'average_score': np.mean([combo['score'] for combo in combinations]),
            'score_distribution': {
                'min': min([combo['score'] for combo in combinations]),
                'max': max([combo['score'] for combo in combinations]),
                'std': np.std([combo['score'] for combo in combinations])
            },
            'most_selected_boules': dict(boules_counter.most_common(10)),
            'chance_distribution': dict(chances_counter),
            'strategies_distribution': Counter([combo['method'].split('_')[0] for combo in combinations])
        }