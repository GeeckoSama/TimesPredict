"""
Analyseur statistique spécialisé pour les patterns du loto français
Détecte les tendances, cycles, et anomalies dans les tirages historiques
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import scipy.stats as stats
from datetime import datetime, timedelta


class LotoStatAnalyzer:
    """
    Analyseur statistique avancé pour les données de loto
    Recherche patterns, tendances, et biais potentiels
    """
    
    def __init__(self, processed_data: pd.DataFrame):
        """
        Initialise l'analyseur avec les données traitées
        
        Args:
            processed_data: DataFrame avec les tirages traités
        """
        self.data = processed_data.copy()
        self.boules_cols = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']
        self.analysis_results = {}
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Exécute l'analyse statistique complète"""
        print("🔍 ANALYSE STATISTIQUE COMPLÈTE DU LOTO")
        print("=" * 50)
        
        # 1. Analyse des fréquences
        print("📊 Analyse des fréquences...")
        freq_analysis = self.analyze_frequencies()
        
        # 2. Tests de randomité
        print("🎲 Tests de randomité...")
        randomness_tests = self.test_randomness()
        
        # 3. Analyse temporelle
        print("📅 Analyse des patterns temporels...")
        temporal_analysis = self.analyze_temporal_patterns()
        
        # 4. Analyse des corrélations
        print("🔗 Analyse des corrélations...")
        correlation_analysis = self.analyze_correlations()
        
        # 5. Détection de cycles
        print("🌀 Détection de cycles...")
        cycle_analysis = self.detect_cycles()
        
        # 6. Analyse des séquences
        print("🔢 Analyse des séquences...")
        sequence_analysis = self.analyze_sequences()
        
        # 7. Tests statistiques avancés
        print("🧮 Tests statistiques...")
        statistical_tests = self.advanced_statistical_tests()
        
        # Consolidation des résultats
        self.analysis_results = {
            'frequencies': freq_analysis,
            'randomness': randomness_tests,
            'temporal': temporal_analysis,
            'correlations': correlation_analysis,
            'cycles': cycle_analysis,
            'sequences': sequence_analysis,
            'statistical_tests': statistical_tests,
            'summary': self._generate_summary()
        }
        
        print("✅ Analyse complète terminée")
        return self.analysis_results
    
    def analyze_frequencies(self) -> Dict[str, Any]:
        """Analyse détaillée des fréquences de sortie"""
        print("   Calcul des fréquences de sortie...")
        
        # Fréquences des boules (1-49)
        all_boules = pd.concat([self.data[col] for col in self.boules_cols])
        boules_freq = all_boules.value_counts().sort_index()
        
        # Fréquences du numéro chance (1-10)
        chance_freq = self.data['numero_chance'].value_counts().sort_index()
        
        # Statistiques de fréquence
        freq_mean = boules_freq.mean()
        freq_std = boules_freq.std()
        
        # Classification hot/cold
        hot_threshold = freq_mean + 0.5 * freq_std
        cold_threshold = freq_mean - 0.5 * freq_std
        
        hot_numbers = boules_freq[boules_freq >= hot_threshold].index.tolist()
        cold_numbers = boules_freq[boules_freq <= cold_threshold].index.tolist()
        
        # Test du chi-carré pour uniformité des boules
        expected_freq = len(all_boules) / 49  # Fréquence attendue si uniforme
        chi2_boules, p_value_boules = stats.chisquare(boules_freq.values)
        
        # Test du chi-carré pour uniformité du numéro chance
        expected_chance = len(self.data) / 10
        chi2_chance, p_value_chance = stats.chisquare(chance_freq.values)
        
        return {
            'boules_frequencies': boules_freq.to_dict(),
            'chance_frequencies': chance_freq.to_dict(),
            'statistics': {
                'boules_mean': freq_mean,
                'boules_std': freq_std,
                'most_frequent': boules_freq.idxmax(),
                'least_frequent': boules_freq.idxmin(),
                'frequency_range': boules_freq.max() - boules_freq.min()
            },
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers,
            'chi2_tests': {
                'boules': {'chi2': chi2_boules, 'p_value': p_value_boules},
                'chance': {'chi2': chi2_chance, 'p_value': p_value_chance}
            }
        }
    
    def test_randomness(self) -> Dict[str, Any]:
        """Tests de randomité sur les séries de tirages"""
        print("   Tests de randomité...")
        
        results = {}
        
        for col in self.boules_cols + ['numero_chance']:
            series = self.data[col].values
            
            # Test des runs (séquences consécutives)
            runs_test = self._runs_test(series)
            
            # Test de Kolmogorov-Smirnov contre distribution uniforme
            if col == 'numero_chance':
                uniform_dist = stats.uniform(loc=1, scale=9)  # 1-10
            else:
                uniform_dist = stats.uniform(loc=1, scale=48)  # 1-49
            
            ks_stat, ks_p = stats.kstest(series, uniform_dist.cdf)
            
            # Autocorrélation (dépendance temporelle)
            autocorr = self._calculate_autocorrelation(series, max_lag=20)
            
            results[col] = {
                'runs_test': runs_test,
                'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p},
                'autocorrelation': autocorr
            }
        
        return results
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyse les patterns liés au temps"""
        print("   Analyse des patterns temporels...")
        
        # Conversion des dates
        self.data['datetime'] = pd.to_datetime(self.data['date_de_tirage'], format='%d/%m/%Y')
        self.data['day_of_week'] = self.data['datetime'].dt.dayofweek
        self.data['month'] = self.data['datetime'].dt.month
        self.data['year'] = self.data['datetime'].dt.year
        
        temporal_patterns = {}
        
        # Patterns par jour de la semaine
        dow_patterns = {}
        for dow in range(7):
            dow_data = self.data[self.data['day_of_week'] == dow]
            if len(dow_data) > 0:
                dow_boules = pd.concat([dow_data[col] for col in self.boules_cols])
                dow_patterns[dow] = {
                    'count': len(dow_data),
                    'avg_sum': dow_data['somme_boules'].mean(),
                    'most_common': dow_boules.mode().tolist()[:5] if not dow_boules.empty else []
                }
        
        # Patterns par mois
        monthly_patterns = {}
        for month in range(1, 13):
            month_data = self.data[self.data['month'] == month]
            if len(month_data) > 0:
                month_boules = pd.concat([month_data[col] for col in self.boules_cols])
                monthly_patterns[month] = {
                    'count': len(month_data),
                    'avg_sum': month_data['somme_boules'].mean(),
                    'most_common': month_boules.mode().tolist()[:5] if not month_boules.empty else []
                }
        
        # Évolution dans le temps
        time_evolution = {}
        for year in self.data['year'].unique():
            year_data = self.data[self.data['year'] == year]
            if len(year_data) > 0:
                year_boules = pd.concat([year_data[col] for col in self.boules_cols])
                time_evolution[year] = {
                    'count': len(year_data),
                    'avg_sum': year_data['somme_boules'].mean(),
                    'top_numbers': year_boules.value_counts().head().to_dict()
                }
        
        return {
            'day_of_week_patterns': dow_patterns,
            'monthly_patterns': monthly_patterns,
            'yearly_evolution': time_evolution
        }
    
    def analyze_correlations(self) -> Dict[str, Any]:
        """Analyse les corrélations entre boules et avec le temps"""
        print("   Calcul des corrélations...")
        
        # Matrice de corrélation entre boules
        boules_corr = self.data[self.boules_cols].corr()
        
        # Corrélations avec les statistiques dérivées
        extended_cols = self.boules_cols + ['numero_chance', 'somme_boules', 'nb_pairs']
        extended_corr = self.data[extended_cols].corr()
        
        # Corrélations temporelles
        if 'datetime' in self.data.columns:
            self.data['days_since_start'] = (self.data['datetime'] - self.data['datetime'].min()).dt.days
            temporal_corr = self.data[extended_cols + ['days_since_start']].corr()['days_since_start'].drop('days_since_start')
        else:
            temporal_corr = pd.Series(dtype=float)
        
        # Détection des corrélations significatives
        significant_corr = []
        for i in range(len(self.boules_cols)):
            for j in range(i+1, len(self.boules_cols)):
                corr_val = boules_corr.iloc[i, j]
                if abs(corr_val) > 0.1:  # Seuil de significativité
                    significant_corr.append({
                        'boule_1': self.boules_cols[i],
                        'boule_2': self.boules_cols[j],
                        'correlation': corr_val
                    })
        
        return {
            'boules_correlation_matrix': boules_corr.to_dict(),
            'extended_correlation_matrix': extended_corr.to_dict(),
            'temporal_correlations': temporal_corr.to_dict(),
            'significant_correlations': significant_corr
        }
    
    def detect_cycles(self) -> Dict[str, Any]:
        """Détecte des cycles potentiels dans les tirages"""
        print("   Détection de cycles...")
        
        cycle_results = {}
        
        for col in self.boules_cols + ['numero_chance', 'somme_boules']:
            series = self.data[col].values
            
            # Analyse spectrale (FFT) pour détecter des périodicités
            fft_result = np.fft.fft(series)
            frequencies = np.fft.fftfreq(len(series))
            
            # Trouver les fréquences dominantes
            power_spectrum = np.abs(fft_result)
            dominant_freqs = frequencies[np.argsort(power_spectrum)[-10:]]  # Top 10
            
            # Convertir en périodes (éviter division par zéro)
            dominant_periods = []
            for freq in dominant_freqs:
                if abs(freq) > 1e-10:  # Éviter les fréquences très proches de zéro
                    period = 1 / abs(freq)
                    if 2 <= period <= len(series) / 4:  # Périodes plausibles
                        dominant_periods.append(period)
            
            cycle_results[col] = {
                'dominant_periods': sorted(dominant_periods),
                'power_spectrum_max': float(power_spectrum.max()),
                'series_length': len(series)
            }
        
        return cycle_results
    
    def analyze_sequences(self) -> Dict[str, Any]:
        """Analyse les séquences et patterns de répétition"""
        print("   Analyse des séquences...")
        
        # Analyse des gaps (intervalles entre sorties du même numéro)
        gaps_analysis = {}
        
        for num in range(1, 50):  # Boules 1-49
            # Positions où ce numéro est sorti
            positions = []
            for idx, row in self.data.iterrows():
                if num in [row[col] for col in self.boules_cols]:
                    positions.append(idx)
            
            if len(positions) > 1:
                gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                gaps_analysis[num] = {
                    'appearances': len(positions),
                    'gaps': gaps,
                    'avg_gap': np.mean(gaps) if gaps else 0,
                    'last_appearance': positions[-1] if positions else -1,
                    'since_last': len(self.data) - 1 - positions[-1] if positions else len(self.data)
                }
        
        # Analyse des séquences consécutives
        consecutive_analysis = self._analyze_consecutive_numbers()
        
        # Patterns de répétition
        repetition_patterns = self._analyze_repetition_patterns()
        
        return {
            'gaps_analysis': gaps_analysis,
            'consecutive_numbers': consecutive_analysis,
            'repetition_patterns': repetition_patterns
        }
    
    def advanced_statistical_tests(self) -> Dict[str, Any]:
        """Tests statistiques avancés"""
        print("   Tests statistiques avancés...")
        
        tests_results = {}
        
        # Test de normalité sur la somme des boules
        shapiro_stat, shapiro_p = stats.shapiro(self.data['somme_boules'])
        
        # Test de Student sur la moyenne
        expected_mean = (1 + 49) * 5 / 2  # Moyenne théorique si tirage uniforme
        t_stat, t_p = stats.ttest_1samp(self.data['somme_boules'], expected_mean)
        
        # Test de Levene pour homogénéité des variances par année
        if 'year' in self.data.columns:
            years = self.data['year'].unique()
            if len(years) > 1:
                year_groups = [self.data[self.data['year'] == year]['somme_boules'].values 
                              for year in years if len(self.data[self.data['year'] == year]) > 10]
                if len(year_groups) > 1:
                    levene_stat, levene_p = stats.levene(*year_groups)
                else:
                    levene_stat, levene_p = None, None
            else:
                levene_stat, levene_p = None, None
        else:
            levene_stat, levene_p = None, None
        
        # Analyse de la parité
        parity_test = self._test_parity_bias()
        
        tests_results = {
            'normality_test': {'shapiro_stat': shapiro_stat, 'p_value': shapiro_p},
            'mean_test': {'t_statistic': t_stat, 'p_value': t_p, 'expected_mean': expected_mean},
            'variance_homogeneity': {'levene_stat': levene_stat, 'p_value': levene_p},
            'parity_bias': parity_test
        }
        
        return tests_results
    
    def _runs_test(self, series: np.ndarray) -> Dict[str, float]:
        """Test des runs pour évaluer la randomité"""
        median_val = np.median(series)
        runs = []
        current_run = 1
        
        for i in range(1, len(series)):
            if (series[i] >= median_val) == (series[i-1] >= median_val):
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        n_runs = len(runs)
        n1 = sum(1 for x in series if x >= median_val)
        n2 = len(series) - n1
        
        # Statistique du test
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        
        if variance_runs > 0:
            z_stat = (n_runs - expected_runs) / np.sqrt(variance_runs)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat, p_value = 0, 1
        
        return {
            'n_runs': n_runs,
            'expected_runs': expected_runs,
            'z_statistic': z_stat,
            'p_value': p_value
        }
    
    def _calculate_autocorrelation(self, series: np.ndarray, max_lag: int = 20) -> Dict[int, float]:
        """Calcule l'autocorrélation pour différents lags"""
        autocorr = {}
        
        for lag in range(1, min(max_lag + 1, len(series) // 4)):
            if lag < len(series):
                corr_coef = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                autocorr[lag] = corr_coef if not np.isnan(corr_coef) else 0.0
        
        return autocorr
    
    def _analyze_consecutive_numbers(self) -> Dict[str, Any]:
        """Analyse les numéros consécutifs dans les tirages"""
        consecutive_counts = []
        
        for _, row in self.data.iterrows():
            boules = sorted([row[col] for col in self.boules_cols])
            consecutive_pairs = 0
            
            for i in range(len(boules) - 1):
                if boules[i + 1] == boules[i] + 1:
                    consecutive_pairs += 1
            
            consecutive_counts.append(consecutive_pairs)
        
        return {
            'average_consecutive_pairs': np.mean(consecutive_counts),
            'max_consecutive_pairs': max(consecutive_counts),
            'distribution': dict(Counter(consecutive_counts))
        }
    
    def _analyze_repetition_patterns(self) -> Dict[str, Any]:
        """Analyse les patterns de répétition entre tirages"""
        repetitions = []
        
        for i in range(1, len(self.data)):
            current_boules = set([self.data.iloc[i][col] for col in self.boules_cols])
            prev_boules = set([self.data.iloc[i-1][col] for col in self.boules_cols])
            
            common = len(current_boules.intersection(prev_boules))
            repetitions.append(common)
        
        return {
            'average_repetitions': np.mean(repetitions) if repetitions else 0,
            'max_repetitions': max(repetitions) if repetitions else 0,
            'distribution': dict(Counter(repetitions)) if repetitions else {}
        }
    
    def _test_parity_bias(self) -> Dict[str, Any]:
        """Test le biais de parité (pairs vs impairs)"""
        parity_counts = self.data['nb_pairs'].value_counts()
        
        # Test binomial: probabilité d'avoir k boules paires sur 5
        expected_probs = [stats.binom.pmf(k, 5, 0.5) for k in range(6)]
        observed_counts = [parity_counts.get(k, 0) for k in range(6)]
        expected_counts = [prob * len(self.data) for prob in expected_probs]
        
        # Test du chi-carré
        chi2_stat, p_value = stats.chisquare(observed_counts, expected_counts)
        
        return {
            'observed_distribution': dict(parity_counts),
            'expected_distribution': {k: expected_counts[k] for k in range(6)},
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'average_pairs': self.data['nb_pairs'].mean()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Génère un résumé des analyses"""
        summary = {
            'total_draws': len(self.data),
            'period_analyzed': self._get_date_range(),
            'key_findings': [],
            'statistical_significance': {},
            'randomness_assessment': 'pending'
        }
        
        # Évaluation générale de la randomité
        randomness_scores = []
        
        if 'randomness' in self.analysis_results:
            for col_test in self.analysis_results['randomness'].values():
                if 'kolmogorov_smirnov' in col_test:
                    randomness_scores.append(col_test['kolmogorov_smirnov']['p_value'])
        
        if randomness_scores:
            avg_randomness = np.mean(randomness_scores)
            if avg_randomness > 0.05:
                summary['randomness_assessment'] = 'Conforme à la randomité attendue'
            elif avg_randomness > 0.01:
                summary['randomness_assessment'] = 'Quelques déviations mineures de la randomité'
            else:
                summary['randomness_assessment'] = 'Déviations significatives de la randomité détectées'
        
        return summary
    
    def _get_date_range(self) -> str:
        """Retourne la plage de dates analysée"""
        try:
            if 'datetime' in self.data.columns:
                dates = self.data['datetime']
                return f"{dates.min().strftime('%d/%m/%Y')} → {dates.max().strftime('%d/%m/%Y')}"
            else:
                return "Plage de dates non disponible"
        except:
            return "Erreur dans le calcul de la plage"
    
    def get_prediction_insights(self) -> Dict[str, Any]:
        """Fournit des insights pour améliorer les prédictions"""
        if not self.analysis_results:
            self.run_full_analysis()
        
        insights = {
            'recommended_numbers': [],
            'avoid_numbers': [],
            'temporal_recommendations': {},
            'confidence_factors': {}
        }
        
        # Recommandations basées sur les fréquences
        if 'frequencies' in self.analysis_results:
            freq_data = self.analysis_results['frequencies']
            
            # Numéros à considérer (ni trop chauds, ni trop froids)
            hot_numbers = freq_data.get('hot_numbers', [])
            cold_numbers = freq_data.get('cold_numbers', [])
            
            all_numbers = set(range(1, 50))
            moderate_numbers = list(all_numbers - set(hot_numbers) - set(cold_numbers))
            
            insights['recommended_numbers'] = moderate_numbers[:10]
            insights['avoid_numbers'] = cold_numbers[-5:] if cold_numbers else []
        
        # Facteurs de confiance
        if 'statistical_tests' in self.analysis_results:
            tests = self.analysis_results['statistical_tests']
            
            # Plus les p-values sont élevées, plus le système semble aléatoire
            insights['confidence_factors']['randomness_level'] = 'high' if \
                tests.get('mean_test', {}).get('p_value', 0) > 0.05 else 'low'
        
        return insights