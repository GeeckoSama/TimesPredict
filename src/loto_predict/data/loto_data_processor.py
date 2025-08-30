"""
Processeur de données pour les tirages de loto français
Charge, nettoie et transforme les données historiques du loto
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path


class LotoDataProcessor:
    """Processeur spécialisé pour les données de loto français"""
    
    def __init__(self, data_path: str):
        """
        Initialise le processeur avec le chemin vers les données
        
        Args:
            data_path: Chemin vers le fichier CSV des tirages loto
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
        self.time_series = {}
        
    def load_data(self) -> pd.DataFrame:
        """Charge les données brutes du fichier CSV"""
        print(f"🔄 Chargement des données loto depuis: {self.data_path.name}")
        
        try:
            # Chargement avec séparateur français
            self.raw_data = pd.read_csv(
                self.data_path, 
                sep=';', 
                encoding='utf-8'
            )
            print(f"✅ {len(self.raw_data)} tirages chargés")
            print(f"   Période: {self._get_date_range()}")
            return self.raw_data
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            raise
    
    def _get_date_range(self) -> str:
        """Retourne la plage de dates des tirages"""
        if self.raw_data is None:
            return "Non chargé"
            
        try:
            dates = pd.to_datetime(self.raw_data['date_de_tirage'], format='%d/%m/%Y')
            return f"{dates.min().strftime('%d/%m/%Y')} → {dates.max().strftime('%d/%m/%Y')}"
        except:
            return "Dates invalides"
    
    def process_data(self) -> Dict[str, any]:
        """Traite et nettoie les données loto"""
        if self.raw_data is None:
            raise ValueError("Données non chargées. Appelez load_data() d'abord.")
            
        print("🔧 Traitement des données loto...")
        
        # Nettoyage et transformation
        processed = self.raw_data.copy()
        
        # 1. Conversion des dates
        processed['date'] = pd.to_datetime(processed['date_de_tirage'], format='%d/%m/%Y')
        processed = processed.sort_values('date')
        
        # 2. Extraction des boules principales (1-49)
        boules_cols = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']
        processed[boules_cols] = processed[boules_cols].astype(int)
        
        # 3. Extraction numéro chance (1-10)  
        processed['numero_chance'] = processed['numero_chance'].astype(int)
        
        # 4. Création de statistiques additionnelles
        processed['somme_boules'] = processed[boules_cols].sum(axis=1)
        processed['moyenne_boules'] = processed[boules_cols].mean(axis=1)
        processed['min_boule'] = processed[boules_cols].min(axis=1)
        processed['max_boule'] = processed[boules_cols].max(axis=1)
        processed['ecart_min_max'] = processed['max_boule'] - processed['min_boule']
        
        # 5. Patterns temporels
        processed['jour_semaine'] = processed['date'].dt.dayofweek
        processed['mois'] = processed['date'].dt.month
        processed['annee'] = processed['date'].dt.year
        
        # 6. Analyse des parités (pair/impair)
        for i, col in enumerate(boules_cols, 1):
            processed[f'boule_{i}_parite'] = processed[col] % 2
        processed['nb_pairs'] = processed[[f'boule_{i}_parite' for i in range(1,6)]].sum(axis=1)
        processed['nb_impairs'] = 5 - processed['nb_pairs']
        
        # 7. Analyse des déciles (1-10, 11-20, etc.)
        for i, col in enumerate(boules_cols, 1):
            processed[f'boule_{i}_decile'] = ((processed[col] - 1) // 10) + 1
            
        self.processed_data = processed
        
        print(f"✅ Traitement terminé:")
        print(f"   {len(processed)} tirages traités")
        print(f"   {processed.columns.size} colonnes générées")
        
        return {
            'data': processed,
            'stats': self._generate_basic_stats()
        }
    
    def _generate_basic_stats(self) -> Dict[str, any]:
        """Génère des statistiques de base sur les données"""
        if self.processed_data is None:
            return {}
            
        boules_cols = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']
        
        stats = {
            'nb_tirages': len(self.processed_data),
            'periode': self._get_date_range(),
            'somme_moyenne': self.processed_data['somme_boules'].mean(),
            'somme_std': self.processed_data['somme_boules'].std(),
            'boules_frequences': {},
            'chance_frequences': self.processed_data['numero_chance'].value_counts().to_dict(),
            'jours_semaine': self.processed_data['jour_semaine'].value_counts().to_dict()
        }
        
        # Fréquences des boules 1-49
        all_boules = pd.concat([self.processed_data[col] for col in boules_cols])
        stats['boules_frequences'] = all_boules.value_counts().to_dict()
        
        return stats
    
    def create_time_series(self) -> Dict[str, np.ndarray]:
        """Crée les séries temporelles pour TimesFM"""
        if self.processed_data is None:
            raise ValueError("Données non traitées. Appelez process_data() d'abord.")
            
        print("📊 Création des séries temporelles pour TimesFM...")
        
        data = self.processed_data.sort_values('date')
        
        # Stratégie: Chaque boule est une série temporelle continue
        self.time_series = {
            'boule_1': data['boule_1'].values.astype(float),
            'boule_2': data['boule_2'].values.astype(float), 
            'boule_3': data['boule_3'].values.astype(float),
            'boule_4': data['boule_4'].values.astype(float),
            'boule_5': data['boule_5'].values.astype(float),
            'numero_chance': data['numero_chance'].values.astype(float),
            
            # Séries auxiliaires pour enrichir l'analyse
            'somme_boules': data['somme_boules'].values.astype(float),
            'moyenne_boules': data['moyenne_boules'].values.astype(float),
            'nb_pairs': data['nb_pairs'].values.astype(float),
            'ecart_min_max': data['ecart_min_max'].values.astype(float)
        }
        
        # Métadonnées temporelles
        self.time_series['dates'] = data['date'].values
        self.time_series['jours_semaine'] = data['jour_semaine'].values.astype(float)
        
        print(f"✅ {len(self.time_series)} séries temporelles créées")
        for name, series in self.time_series.items():
            if name != 'dates':
                print(f"   {name}: {len(series)} points, μ={series.mean():.1f}, σ={series.std():.1f}")
        
        return self.time_series
    
    def get_recent_context(self, context_length: int = 100) -> Dict[str, np.ndarray]:
        """Retourne les derniers points pour utilisation comme contexte TimesFM"""
        if not self.time_series:
            self.create_time_series()
            
        recent_context = {}
        for name, series in self.time_series.items():
            if name != 'dates' and isinstance(series, np.ndarray):
                # Prendre les derniers context_length points
                recent_context[name] = series[-context_length:] if len(series) > context_length else series
                
        print(f"📋 Contexte récent extrait: {context_length} derniers tirages")
        return recent_context
    
    def split_train_test(self, test_size: int = 110) -> Tuple[Dict, Dict]:
        """Divise les données en train/test pour validation"""
        if not self.time_series:
            self.create_time_series()
            
        train_data = {}
        test_data = {}
        
        for name, series in self.time_series.items():
            if name != 'dates' and isinstance(series, np.ndarray):
                # Train: tout sauf les test_size derniers
                train_data[name] = series[:-test_size] if len(series) > test_size else series[:-1]
                # Test: les test_size derniers
                test_data[name] = series[-test_size:]
                
        print(f"🔄 Division train/test:")
        print(f"   Train: {len(train_data['boule_1'])} tirages")  
        print(f"   Test: {len(test_data['boule_1'])} tirages")
        
        return train_data, test_data
    
    def get_last_combination(self) -> Tuple[List[int], int]:
        """Retourne la dernière combinaison tirée"""
        if self.processed_data is None:
            raise ValueError("Données non traitées.")
            
        last_row = self.processed_data.iloc[-1]
        boules = [int(last_row[f'boule_{i}']) for i in range(1, 6)]
        boules.sort()  # Tri croissant traditionnel
        chance = int(last_row['numero_chance'])
        
        return boules, chance
    
    def analyze_patterns(self) -> Dict[str, any]:
        """Analyse les patterns et tendances dans les données"""
        if self.processed_data is None:
            raise ValueError("Données non traitées.")
            
        print("🔍 Analyse des patterns loto...")
        
        data = self.processed_data
        patterns = {}
        
        # 1. Fréquences des boules (Hot/Cold numbers)
        boules_cols = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']
        all_boules = pd.concat([data[col] for col in boules_cols])
        freq_boules = all_boules.value_counts()
        
        patterns['boules_chaudes'] = freq_boules.head(10).to_dict()  # 10 plus fréquentes
        patterns['boules_froides'] = freq_boules.tail(10).to_dict()  # 10 moins fréquentes
        
        # 2. Patterns temporels
        patterns['repartition_jours'] = data['jour_semaine'].value_counts().to_dict()
        patterns['repartition_mois'] = data['mois'].value_counts().to_dict()
        
        # 3. Analyse parité
        patterns['moyenne_pairs'] = data['nb_pairs'].mean()
        patterns['repartition_parite'] = data['nb_pairs'].value_counts().to_dict()
        
        # 4. Analyse sommes
        patterns['somme_moyenne'] = data['somme_boules'].mean()
        patterns['somme_std'] = data['somme_boules'].std()
        patterns['somme_min'] = data['somme_boules'].min()
        patterns['somme_max'] = data['somme_boules'].max()
        
        # 5. Numéro chance
        patterns['chance_frequences'] = data['numero_chance'].value_counts().to_dict()
        
        # 6. Écarts et spreads
        patterns['ecart_moyen'] = data['ecart_min_max'].mean()
        patterns['ecart_std'] = data['ecart_min_max'].std()
        
        print(f"✅ Analyse des patterns terminée")
        print(f"   Boules les plus chaudes: {list(patterns['boules_chaudes'].keys())[:5]}")
        print(f"   Somme moyenne: {patterns['somme_moyenne']:.1f} ± {patterns['somme_std']:.1f}")
        print(f"   Parité moyenne: {patterns['moyenne_pairs']:.1f} boules paires")
        
        return patterns
    
    def get_summary(self) -> Dict[str, any]:
        """Retourne un résumé complet des données traitées"""
        if self.processed_data is None:
            return {"status": "Données non traitées"}
            
        return {
            "nb_tirages": len(self.processed_data),
            "periode": self._get_date_range(),
            "derniere_combinaison": self.get_last_combination(),
            "series_disponibles": list(self.time_series.keys()) if self.time_series else [],
            "stats_basiques": self._generate_basic_stats()
        }