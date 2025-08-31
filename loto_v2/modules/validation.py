"""
Validation Module - Séries de prédictions et analyses Loto V2
Génère N prédictions et analyse les fréquences d'apparition
"""

from collections import Counter
from typing import Dict, List, Any
from .prediction import LotoPredictor
from .stats import LotoStatsAnalyzer
from .progress import ProgressBar, UnifiedProgressBar


class LotoValidator:
    """Générateur de séries de prédictions avec analyse"""
    
    def __init__(self, data_file: str = "../data/raw/loto_complet_fusionne.csv"):
        self.predictor = LotoPredictor(data_file)
        self.stats_analyzer = LotoStatsAnalyzer(data_file)
        
    def generate_prediction_series(self, n_predictions: int = 100, use_timesfm: bool = True) -> Dict[str, Any]:
        """Génère N prédictions et analyse les fréquences"""
        predictions = []
        boules_counter = Counter()
        chance_counter = Counter()
        method_counter = Counter()
        confidence_scores = []
        
        # Chargement initial (une seule fois)
        initial_load_done = False
        
        # Barre de progression unifiée pour les prédictions multiples
        progress = UnifiedProgressBar(n_predictions, "🎲 Génération prédictions")
        
        # Génération des prédictions avec suivi unifié
        for i in range(n_predictions):
            current = i + 1
            progress.set_step(current, f"Prédiction {current}/{n_predictions}")
            
            # Première prédiction : chargement initial avec messages
            if not initial_load_done:
                if use_timesfm:
                    pred = self.predictor.predict_with_timesfm()
                else:
                    pred = self.predictor.predict_single_combination()
                initial_load_done = True
            else:
                # Prédictions suivantes : mode silencieux
                import io
                import sys
                from contextlib import redirect_stdout, redirect_stderr
                
                # Capturer tous les print() et messages
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    if use_timesfm:
                        pred = self.predictor.predict_with_timesfm()
                    else:
                        pred = self.predictor.predict_single_combination()
            
            predictions.append(pred)
            
            # Comptage des fréquences
            for boule in pred["boules"]:
                boules_counter[boule] += 1
            
            chance_counter[pred["chance"]] += 1
            method_counter[pred["method"]] += 1
            confidence_scores.append(pred["confidence"])
        
        progress.finish("Génération terminée")
        
        # Analyse des résultats
        analysis = self._analyze_prediction_series(
            predictions, boules_counter, chance_counter, 
            method_counter, confidence_scores, n_predictions
        )
        
        return {
            "predictions": predictions,
            "analysis": analysis,
            "metadata": {
                "n_predictions": n_predictions,
                "used_timesfm": use_timesfm
            }
        }
    
    def _analyze_prediction_series(self, predictions: List[Dict], boules_counter: Counter, 
                                 chance_counter: Counter, method_counter: Counter,
                                 confidence_scores: List[float], n_predictions: int) -> Dict[str, Any]:
        """Analyse complète des résultats de prédiction"""
        
        # Top prédictions par fréquence
        most_frequent_boules = boules_counter.most_common(10)
        most_frequent_chances = chance_counter.most_common(10)
        
        # Combinaison optimale basée sur fréquences
        optimal_boules = [boule for boule, _ in most_frequent_boules[:5]]
        optimal_chance = most_frequent_chances[0][0] if most_frequent_chances else 1
        
        # Statistiques de confiance
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Distribution des méthodes utilisées
        method_distribution = dict(method_counter)
        
        # Comparaison avec historique
        historical_comparison = self._compare_with_historical(boules_counter, chance_counter)
        
        return {
            "optimal_combination": {
                "boules": sorted(optimal_boules),
                "chance": optimal_chance,
                "confidence": avg_confidence
            },
            "frequency_analysis": {
                "most_frequent_boules": most_frequent_boules,
                "most_frequent_chances": most_frequent_chances,
                "boules_distribution": dict(boules_counter),
                "chance_distribution": dict(chance_counter)
            },
            "performance_metrics": {
                "average_confidence": round(avg_confidence, 3),
                "confidence_range": [min(confidence_scores), max(confidence_scores)],
                "method_distribution": method_distribution
            },
            "historical_comparison": historical_comparison
        }
    
    def _compare_with_historical(self, pred_boules: Counter, pred_chances: Counter) -> Dict[str, Any]:
        """Compare les prédictions avec les fréquences historiques"""
        try:
            historical_weights = self.stats_analyzer.get_probability_weights()
            
            if not historical_weights or not historical_weights.get("boules"):
                return {"status": "no_historical_data"}
            
            hist_boules = historical_weights["boules"]
            hist_chances = historical_weights["chance"]
            
            # Corrélation entre prédictions et historique
            boules_correlation = self._calculate_correlation(pred_boules, hist_boules)
            chances_correlation = self._calculate_correlation(pred_chances, hist_chances)
            
            # Écarts significatifs
            boules_deviations = self._find_significant_deviations(pred_boules, hist_boules)
            
            return {
                "status": "comparison_available",
                "correlations": {
                    "boules": round(boules_correlation, 3),
                    "chances": round(chances_correlation, 3)
                },
                "significant_deviations": boules_deviations,
                "alignment_score": round((boules_correlation + chances_correlation) / 2, 3)
            }
            
        except Exception as e:
            print(f"⚠️  Erreur comparaison historique: {e}")
            return {"status": "comparison_error"}
    
    def _calculate_correlation(self, predictions: Counter, historical: Dict[int, float]) -> float:
        """Calcule la corrélation entre prédictions et données historiques"""
        if not predictions or not historical:
            return 0.0
        
        # Convertir en fréquences normalisées
        total_pred = sum(predictions.values())
        pred_freqs = {k: v/total_pred for k, v in predictions.items()}
        
        # Corrélation simple basée sur le rang
        common_keys = set(pred_freqs.keys()) & set(historical.keys())
        if len(common_keys) < 2:
            return 0.0
        
        # Calcul de corrélation de Pearson simplifiée
        pred_values = [pred_freqs[k] for k in common_keys]
        hist_values = [historical[k] for k in common_keys]
        
        if len(pred_values) == 0:
            return 0.0
        
        # Corrélation approximative
        pred_mean = sum(pred_values) / len(pred_values)
        hist_mean = sum(hist_values) / len(hist_values)
        
        numerator = sum((p - pred_mean) * (h - hist_mean) for p, h in zip(pred_values, hist_values))
        pred_var = sum((p - pred_mean) ** 2 for p in pred_values)
        hist_var = sum((h - hist_mean) ** 2 for h in hist_values)
        
        if pred_var == 0 or hist_var == 0:
            return 0.0
        
        correlation = numerator / (pred_var * hist_var) ** 0.5
        return max(-1.0, min(1.0, correlation))
    
    def _find_significant_deviations(self, predictions: Counter, historical: Dict[int, float]) -> List[Dict[str, Any]]:
        """Trouve les écarts significatifs entre prédictions et historique"""
        deviations = []
        
        total_pred = sum(predictions.values())
        
        for number in range(1, 50):  # Toutes les boules possibles
            pred_freq = predictions.get(number, 0) / total_pred if total_pred > 0 else 0
            hist_freq = historical.get(number, 0)
            
            if hist_freq > 0:  # Éviter division par zéro
                deviation_ratio = (pred_freq - hist_freq) / hist_freq
                
                # Seuil de signification : 50% d'écart
                if abs(deviation_ratio) > 0.5:
                    deviations.append({
                        "number": number,
                        "predicted_freq": round(pred_freq, 4),
                        "historical_freq": round(hist_freq, 4),
                        "deviation_ratio": round(deviation_ratio, 3),
                        "type": "over_predicted" if deviation_ratio > 0 else "under_predicted"
                    })
        
        # Trier par amplitude de déviation
        deviations.sort(key=lambda x: abs(x["deviation_ratio"]), reverse=True)
        return deviations[:10]  # Top 10 des déviations
    
    def display_series_summary(self, results: Dict[str, Any]):
        """Affiche un résumé des résultats de série"""
        analysis = results["analysis"]
        metadata = results["metadata"]
        
        print(f"\n📊 SÉRIE DE {metadata['n_predictions']} PRÉDICTIONS")
        print(f"🔧 Méthode: {'TimesFM + Pondération' if metadata['used_timesfm'] else 'Pondération seule'}")
        
        # Combinaison optimale
        optimal = analysis["optimal_combination"]
        print(f"\n🎯 COMBINAISON OPTIMALE (la plus fréquente):")
        print(f"   Boules: {optimal['boules']}")
        print(f"   Chance: {optimal['chance']}")
        print(f"   Confiance: {optimal['confidence']:.1%}")
        
        # Top fréquences
        print(f"\n🔥 Top 5 boules les plus prédites:")
        for boule, count in analysis["frequency_analysis"]["most_frequent_boules"][:5]:
            freq_pct = count / metadata['n_predictions'] * 100
            print(f"   {boule:2d}: {count:3d} fois ({freq_pct:.1f}%)")
        
        # Comparaison historique
        hist_comp = analysis["historical_comparison"]
        if hist_comp["status"] == "comparison_available":
            print(f"\n📈 Alignement historique: {hist_comp['alignment_score']:.1%}")
            
            if hist_comp["significant_deviations"]:
                print("⚠️  Écarts significatifs détectés:")
                for dev in hist_comp["significant_deviations"][:3]:
                    print(f"   {dev['number']:2d}: {dev['deviation_ratio']:+.0%} vs historique")
        
        # Métriques de performance
        perf = analysis["performance_metrics"]
        print(f"\n🎲 Confiance moyenne: {perf['average_confidence']:.1%}")
        
        if perf["method_distribution"]:
            print("📋 Méthodes utilisées:")
            for method, count in perf["method_distribution"].items():
                print(f"   {method}: {count} fois")