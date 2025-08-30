"""
Validateur de performance pour les prédictions TimesFM
Effectue des tests de backtest et évalue la précision des modèles
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import warnings

warnings.filterwarnings('ignore')


class BacktestValidator:
    """
    Validateur de performance pour évaluer les prédictions TimesFM
    Utilise des méthodes de backtest sur les données historiques
    """
    
    def __init__(self, historical_data: pd.DataFrame):
        """
        Initialise le validateur avec les données historiques
        
        Args:
            historical_data: DataFrame avec les tirages historiques
        """
        self.historical_data = historical_data
        self.validation_results = None
        
        # Colonnes attendues
        self.boules_cols = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']
        self.chance_col = 'numero_chance'
    
    def run_backtest(self, 
                    predictor,
                    test_period_days: int = 90,
                    prediction_frequency: int = 7) -> Dict[str, Any]:
        """
        Exécute un backtest complet sur une période donnée
        
        Args:
            predictor: Instance du prédicteur TimesFM à tester
            test_period_days: Nombre de tirages à utiliser pour le test
            prediction_frequency: Fréquence des prédictions (tous les N tirages)
            
        Returns:
            Dict avec les résultats détaillés du backtest
        """
        print(f"📊 Lancement du backtest sur {test_period_days} tirages")
        print(f"🔄 Modèle testé: {predictor.__class__.__name__}")
        
        # Vérifier la quantité de données disponibles
        if len(self.historical_data) < test_period_days + 100:
            print(f"❌ Pas assez de données pour le backtest ({len(self.historical_data)} tirages disponibles)")
            return {'error': 'Insufficient data'}
        
        train_data = self.historical_data.iloc[:-test_period_days].copy()
        test_data = self.historical_data.iloc[-test_period_days:].copy()
        
        print(f"   Données d'entraînement: {len(train_data)} tirages")
        print(f"   Données de test: {len(test_data)} tirages")
        print(f"   Fréquence de prédiction: tous les {prediction_frequency} tirages")
        
        backtest_results = []
        prediction_points = list(range(0, len(test_data), prediction_frequency))
        
        for i, test_idx in enumerate(prediction_points):
            if test_idx >= len(test_data) - 1:  # S'assurer qu'il y a au moins 1 tirage à prédire
                break
                
            print(f"\n🎯 Prédiction {i+1}/{len(prediction_points)-1} (tirage {test_idx+1})")
            
            # Données d'entraînement jusqu'à ce point
            current_train = pd.concat([train_data, test_data.iloc[:test_idx]]) if test_idx > 0 else train_data
            
            # Tirage réel à prédire
            actual_draw = test_data.iloc[test_idx]
            actual_boules = sorted([actual_draw[col] for col in self.boules_cols])
            actual_chance = actual_draw['numero_chance']
            
            try:
                # Créer les séries temporelles pour l'entraînement
                time_series = self._create_time_series_for_backtest(current_train)
                
                # Faire la prédiction
                prediction_result = predictor.predict_next_combination(time_series)
                
                # Évaluer la prédiction
                evaluation = self._evaluate_single_prediction(
                    prediction_result, actual_boules, actual_chance, test_idx
                )
                
                evaluation['train_size'] = len(current_train)
                evaluation['actual_combination'] = {
                    'boules': actual_boules,
                    'numero_chance': actual_chance
                }
                
                backtest_results.append(evaluation)
                
                print(f"   Réel: {actual_boules} + {actual_chance}")
                print(f"   Prédit: {prediction_result['combination']['boules']} + {prediction_result['combination']['numero_chance']}")
                print(f"   Score: {evaluation['accuracy_metrics']['total_accuracy']:.1%}")
                
            except Exception as e:
                print(f"   ❌ Erreur prédiction: {e}")
                error_eval = {
                    'test_index': test_idx,
                    'prediction_failed': True,
                    'error': str(e),
                    'actual_combination': {
                        'boules': actual_boules,
                        'numero_chance': actual_chance
                    }
                }
                backtest_results.append(error_eval)
        
        # Compiler les résultats globaux
        overall_results = self._compile_backtest_results(backtest_results)
        
        self.validation_results = {
            'backtest_config': {
                'test_period_days': test_period_days,
                'prediction_frequency': prediction_frequency,
                'total_predictions': len(backtest_results)
            },
            'individual_results': backtest_results,
            'overall_performance': overall_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self._print_backtest_summary(overall_results)
        
        return self.validation_results
    
    def evaluate_combination_batch(self, 
                                  predicted_combinations: List[Dict[str, Any]], 
                                  actual_draws: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Évalue un lot de combinaisons prédites contre des tirages réels"""
        
        print(f"📊 Évaluation de {len(predicted_combinations)} prédictions vs {len(actual_draws)} tirages réels")
        
        batch_results = []
        
        for i, (pred, actual) in enumerate(zip(predicted_combinations, actual_draws)):
            actual_boules = sorted([actual[col] for col in self.boules_cols])
            actual_chance = actual['numero_chance']
            
            evaluation = self._evaluate_single_prediction(pred, actual_boules, actual_chance, i)
            batch_results.append(evaluation)
        
        return self._compile_backtest_results(batch_results)
    
    def _create_time_series_for_backtest(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Crée les séries temporelles pour le backtest"""
        
        # Trier par date pour s'assurer de l'ordre chronologique
        sorted_data = data.sort_values('date') if 'date' in data.columns else data
        
        time_series = {
            'boule_1': sorted_data['boule_1'].values.astype(float),
            'boule_2': sorted_data['boule_2'].values.astype(float),
            'boule_3': sorted_data['boule_3'].values.astype(float),
            'boule_4': sorted_data['boule_4'].values.astype(float),
            'boule_5': sorted_data['boule_5'].values.astype(float),
            'numero_chance': sorted_data['numero_chance'].values.astype(float)
        }
        
        return time_series
    
    def _evaluate_single_prediction(self, 
                                   prediction_result: Dict[str, Any], 
                                   actual_boules: List[int], 
                                   actual_chance: int,
                                   test_index: int) -> Dict[str, Any]:
        """Évalue une prédiction individuelle"""
        
        if 'combination' not in prediction_result:
            return {
                'test_index': test_index,
                'prediction_failed': True,
                'error': 'No combination in prediction result'
            }
        
        pred_combo = prediction_result['combination']
        pred_boules = pred_combo['boules']
        pred_chance = pred_combo['numero_chance']
        
        # Calcul des métriques de précision
        accuracy_metrics = self._calculate_accuracy_metrics(
            pred_boules, pred_chance, actual_boules, actual_chance
        )
        
        # Analyse des écarts
        proximity_analysis = self._analyze_number_proximity(
            pred_boules, actual_boules
        )
        
        # Métriques spécifiques au loto
        loto_metrics = self._calculate_loto_specific_metrics(
            pred_boules, pred_chance, actual_boules, actual_chance
        )
        
        return {
            'test_index': test_index,
            'predicted_combination': {'boules': pred_boules, 'numero_chance': pred_chance},
            'actual_combination': {'boules': actual_boules, 'numero_chance': actual_chance},
            'accuracy_metrics': accuracy_metrics,
            'proximity_analysis': proximity_analysis,
            'loto_metrics': loto_metrics,
            'prediction_metadata': prediction_result.get('metadata', {})
        }
    
    def _calculate_accuracy_metrics(self, 
                                   pred_boules: List[int], pred_chance: int,
                                   actual_boules: List[int], actual_chance: int) -> Dict[str, float]:
        """Calcule les métriques de précision"""
        
        # Boules exactes
        exact_boules = len(set(pred_boules) & set(actual_boules))
        exact_boules_rate = exact_boules / 5
        
        # Numéro chance exact
        exact_chance = 1 if pred_chance == actual_chance else 0
        
        # Score total (pondéré)
        total_accuracy = (exact_boules_rate * 0.8) + (exact_chance * 0.2)
        
        return {
            'exact_boules': exact_boules,
            'exact_boules_rate': exact_boules_rate,
            'exact_chance': exact_chance,
            'total_accuracy': total_accuracy
        }
    
    def _analyze_number_proximity(self, pred_boules: List[int], actual_boules: List[int]) -> Dict[str, Any]:
        """Analyse la proximité des numéros prédits vs réels"""
        
        proximities = []
        
        for pred_num in pred_boules:
            # Distance minimale à un numéro réel
            min_distance = min(abs(pred_num - actual_num) for actual_num in actual_boules)
            proximities.append(min_distance)
        
        # Numéros "proches" (écart <= 3)
        close_numbers = sum(1 for dist in proximities if dist <= 3)
        very_close_numbers = sum(1 for dist in proximities if dist <= 1)
        
        return {
            'min_distances': proximities,
            'average_distance': np.mean(proximities),
            'close_numbers': close_numbers,  # <= 3 de distance
            'very_close_numbers': very_close_numbers,  # <= 1 de distance
            'max_distance': max(proximities)
        }
    
    def _calculate_loto_specific_metrics(self, 
                                        pred_boules: List[int], pred_chance: int,
                                        actual_boules: List[int], actual_chance: int) -> Dict[str, Any]:
        """Calcule des métriques spécifiques au loto français"""
        
        # Rangs de gains simulés (approximation)
        exact_matches = len(set(pred_boules) & set(actual_boules))
        chance_match = pred_chance == actual_chance
        
        # Simulation des rangs de gain du loto français
        gain_rank = None
        if exact_matches == 5 and chance_match:
            gain_rank = 1  # Jackpot
        elif exact_matches == 5:
            gain_rank = 2  # 5 boules sans chance
        elif exact_matches == 4 and chance_match:
            gain_rank = 3  # 4 boules + chance
        elif exact_matches == 4:
            gain_rank = 4  # 4 boules sans chance
        elif exact_matches == 3 and chance_match:
            gain_rank = 5  # 3 boules + chance
        elif exact_matches == 3:
            gain_rank = 6  # 3 boules sans chance
        elif exact_matches == 2 and chance_match:
            gain_rank = 7  # 2 boules + chance
        elif chance_match:
            gain_rank = 8  # Chance seule
        elif exact_matches == 2:
            gain_rank = 9  # 2 boules sans chance
        
        # Analyse des patterns
        pred_sum = sum(pred_boules)
        actual_sum = sum(actual_boules)
        sum_difference = abs(pred_sum - actual_sum)
        
        pred_pairs = sum(1 for b in pred_boules if b % 2 == 0)
        actual_pairs = sum(1 for b in actual_boules if b % 2 == 0)
        parity_match = pred_pairs == actual_pairs
        
        return {
            'gain_rank': gain_rank,
            'would_win': gain_rank is not None,
            'sum_prediction_accuracy': sum_difference,
            'sum_predicted': pred_sum,
            'sum_actual': actual_sum,
            'parity_match': parity_match,
            'pairs_predicted': pred_pairs,
            'pairs_actual': actual_pairs
        }
    
    def _compile_backtest_results(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile les résultats individuels en statistiques globales"""
        
        successful_predictions = [r for r in individual_results if not r.get('prediction_failed', False)]
        
        if not successful_predictions:
            return {
                'total_predictions': len(individual_results),
                'successful_predictions': 0,
                'success_rate': 0.0,
                'error': 'No successful predictions'
            }
        
        # Métriques de précision globales
        total_exact_boules = sum(r['accuracy_metrics']['exact_boules'] for r in successful_predictions)
        total_exact_chances = sum(r['accuracy_metrics']['exact_chance'] for r in successful_predictions)
        avg_total_accuracy = np.mean([r['accuracy_metrics']['total_accuracy'] for r in successful_predictions])
        
        # Métriques de proximité
        avg_distance = np.mean([r['proximity_analysis']['average_distance'] for r in successful_predictions])
        total_close_numbers = sum(r['proximity_analysis']['close_numbers'] for r in successful_predictions)
        
        # Métriques loto
        winning_predictions = sum(1 for r in successful_predictions if r['loto_metrics']['would_win'])
        gain_rank_distribution = {}
        for r in successful_predictions:
            rank = r['loto_metrics']['gain_rank']
            if rank:
                gain_rank_distribution[rank] = gain_rank_distribution.get(rank, 0) + 1
        
        # Distribution des performances
        exact_boules_distribution = {}
        for i in range(6):  # 0 à 5 boules exactes
            count = sum(1 for r in successful_predictions if r['accuracy_metrics']['exact_boules'] == i)
            exact_boules_distribution[i] = count
        
        return {
            'total_predictions': len(individual_results),
            'successful_predictions': len(successful_predictions),
            'success_rate': len(successful_predictions) / len(individual_results),
            
            'accuracy_summary': {
                'average_exact_boules': total_exact_boules / len(successful_predictions),
                'exact_chance_rate': total_exact_chances / len(successful_predictions),
                'average_total_accuracy': avg_total_accuracy,
                'exact_boules_distribution': exact_boules_distribution
            },
            
            'proximity_summary': {
                'average_distance': avg_distance,
                'close_numbers_rate': total_close_numbers / (len(successful_predictions) * 5)
            },
            
            'loto_performance': {
                'winning_predictions': winning_predictions,
                'win_rate': winning_predictions / len(successful_predictions),
                'gain_rank_distribution': gain_rank_distribution
            },
            
            'statistical_significance': self._calculate_statistical_significance(successful_predictions)
        }
    
    def _calculate_statistical_significance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcule la significativité statistique des résultats"""
        
        if len(results) < 10:  # Pas assez de données pour des tests significatifs
            return {'insufficient_data': True}
        
        # Test si les performances sont meilleures que le hasard
        exact_boules_rates = [r['accuracy_metrics']['exact_boules_rate'] for r in results]
        
        # Probabilité théorique d'avoir k boules exactes par hasard
        # C(5,k) * C(44,5-k) / C(49,5) pour k boules exactes
        from math import comb
        
        expected_0_exact = comb(44, 5) / comb(49, 5)  # ~0.5012
        expected_1_exact = comb(5, 1) * comb(44, 4) / comb(49, 5)  # ~0.4130
        expected_2_exact = comb(5, 2) * comb(44, 3) / comb(49, 5)  # ~0.0815
        
        expected_avg_exact_rate = (0 * expected_0_exact + 
                                   1/5 * expected_1_exact + 
                                   2/5 * expected_2_exact)  # ~0.099 ou 9.9%
        
        observed_avg_exact_rate = np.mean(exact_boules_rates)
        
        # Test t simple
        import scipy.stats as stats
        t_stat, p_value = stats.ttest_1samp(exact_boules_rates, expected_avg_exact_rate)
        
        return {
            'expected_random_performance': expected_avg_exact_rate,
            'observed_performance': observed_avg_exact_rate,
            'improvement_factor': observed_avg_exact_rate / expected_avg_exact_rate,
            't_statistic': t_stat,
            'p_value': p_value,
            'significantly_better': p_value < 0.05 and t_stat > 0
        }
    
    def _print_backtest_summary(self, overall_results: Dict[str, Any]):
        """Affiche un résumé des résultats de backtest"""
        
        print("\n" + "=" * 50)
        print("📈 RÉSUMÉ DU BACKTEST")
        print("=" * 50)
        
        print(f"\n🎯 PERFORMANCES GLOBALES:")
        print(f"   Total prédictions: {overall_results['total_predictions']}")
        print(f"   Prédictions réussies: {overall_results['successful_predictions']}")
        print(f"   Taux de succès: {overall_results['success_rate']:.1%}")
        
        if 'accuracy_summary' in overall_results:
            acc = overall_results['accuracy_summary']
            print(f"\n🎲 PRÉCISION:")
            print(f"   Boules exactes (moyenne): {acc['average_exact_boules']:.2f}/5")
            print(f"   Taux chance exacte: {acc['exact_chance_rate']:.1%}")
            print(f"   Score de précision global: {acc['average_total_accuracy']:.1%}")
            
            print(f"\n📊 DISTRIBUTION BOULES EXACTES:")
            for k, count in acc['exact_boules_distribution'].items():
                rate = count / overall_results['successful_predictions'] if overall_results['successful_predictions'] > 0 else 0
                print(f"   {k} boules exactes: {count} fois ({rate:.1%})")
        
        if 'loto_performance' in overall_results:
            loto = overall_results['loto_performance']
            print(f"\n🏆 PERFORMANCE LOTO:")
            print(f"   Combinaisons gagnantes: {loto['winning_predictions']}")
            print(f"   Taux de gain: {loto['win_rate']:.1%}")
            
            if loto['gain_rank_distribution']:
                print(f"   Répartition des rangs de gain:")
                for rank, count in sorted(loto['gain_rank_distribution'].items()):
                    print(f"     Rang {rank}: {count} fois")
        
        if 'statistical_significance' in overall_results:
            sig = overall_results['statistical_significance']
            if not sig.get('insufficient_data', False):
                print(f"\n🧮 SIGNIFICATIVITÉ STATISTIQUE:")
                print(f"   Performance attendue (hasard): {sig['expected_random_performance']:.1%}")
                print(f"   Performance observée: {sig['observed_performance']:.1%}")
                print(f"   Facteur d'amélioration: {sig['improvement_factor']:.2f}x")
                print(f"   Significativement meilleur: {'✅ OUI' if sig['significantly_better'] else '❌ NON'}")
                print(f"   P-value: {sig['p_value']:.4f}")
    
    def save_results(self, filepath: str):
        """Sauvegarde les résultats de validation"""
        if not self.validation_results:
            print("❌ Aucun résultat à sauvegarder")
            return False
            
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
            print(f"✅ Résultats sauvegardés: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            return False
    
    def load_results(self, filepath: str) -> bool:
        """Charge des résultats de validation sauvegardés"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.validation_results = json.load(f)
            print(f"✅ Résultats chargés: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
            return False
    
    def get_prediction_recommendations(self) -> Dict[str, Any]:
        """Fournit des recommandations basées sur les résultats de validation"""
        if not self.validation_results:
            return {'error': 'No validation results available'}
        
        overall = self.validation_results.get('overall_performance', {})
        
        recommendations = {
            'model_performance': 'unknown',
            'recommended_usage': 'cautious',
            'confidence_level': 'low',
            'suggestions': []
        }
        
        if 'accuracy_summary' in overall:
            acc = overall['accuracy_summary']['average_total_accuracy']
            
            if acc > 0.15:  # > 15% de précision globale
                recommendations['model_performance'] = 'good'
                recommendations['confidence_level'] = 'high'
                recommendations['recommended_usage'] = 'active'
                recommendations['suggestions'].append("Modèle performant, utilisation recommandée")
            elif acc > 0.10:  # > 10% de précision
                recommendations['model_performance'] = 'acceptable'
                recommendations['confidence_level'] = 'medium'
                recommendations['recommended_usage'] = 'moderate'
                recommendations['suggestions'].append("Performance acceptable, utiliser avec prudence")
            else:
                recommendations['model_performance'] = 'poor'
                recommendations['suggestions'].append("Performance faible, améliorer le modèle")
        
        # Ajouter des suggestions spécifiques
        if 'statistical_significance' in overall:
            sig = overall['statistical_significance']
            if sig.get('significantly_better', False):
                recommendations['suggestions'].append("Amélioration statistiquement significative vs hasard")
            else:
                recommendations['suggestions'].append("Pas d'amélioration significative vs hasard détectée")
        
        return recommendations