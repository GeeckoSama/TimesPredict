"""
Stats Module - Analyse des fréquences historiques Loto V2
Calcul et analyse des fréquences d'apparition par chiffre
"""

import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Any
from .storage import LotoStorage
from .progress import ProgressBar, UnifiedProgressBar


class LotoStatsAnalyzer:
    """Analyseur de statistiques pour les tirages loto"""
    
    def __init__(self, data_file: str = "../data/raw/loto_complet_fusionne.csv"):
        self.data_file = data_file
        self.storage = LotoStorage()
        self.df = None
        self.stats_data = None
    
    def load_data(self) -> bool:
        """Charge les données historiques"""
        try:
            progress = ProgressBar(2, "📊 Chargement données")
            progress.update(1)
            
            self.df = pd.read_csv(self.data_file, sep=';')
            progress.set_description(f"📊 {len(self.df)} tirages chargés")
            progress.update(1)
            
            return True
        except Exception as e:
            print(f"❌ Erreur chargement données: {e}")
            return False
    
    def calculate_frequencies(self) -> Dict[str, Any]:
        """Calcule toutes les fréquences d'apparition"""
        if self.df is None:
            if not self.load_data():
                return {}
        
        if self.df is None:  # Double vérification après load_data
            return {}
        
        # Barre de progression unifiée avec 4 étapes
        progress = UnifiedProgressBar(4, "📊 Calcul statistiques")
        
        # Étape 1: Initialisation
        progress.set_step(1, "Initialisation des compteurs")
        boules_freq = Counter()
        chance_freq = Counter()
        
        # Étape 2: Calcul des fréquences
        progress.set_step(2, f"Analyse de {len(self.df)} tirages historiques")
        total_rows = len(self.df)
        
        # Compter fréquences par chiffre avec suivi détaillé
        for idx, (_, row) in enumerate(self.df.iterrows()):
            # Mise à jour du statut tous les 500 tirages
            if idx % 500 == 0 or idx == total_rows - 1:
                progress.update_action(f"Tirage {idx+1}/{total_rows}")
            
            for i in range(1, 6):  # boule_1 à boule_5
                boule = int(row[f'boule_{i}'])
                if 1 <= boule <= 49:
                    boules_freq[boule] += 1
            
            chance = int(row['numero_chance'])
            if 1 <= chance <= 10:
                chance_freq[chance] += 1
        
        # Étape 3: Calcul des probabilités
        progress.set_step(3, "Calcul des probabilités normalisées")
        total_boules = sum(boules_freq.values())
        total_chances = sum(chance_freq.values())
        
        boules_prob = {num: freq/total_boules for num, freq in boules_freq.items()}
        chance_prob = {num: freq/total_chances for num, freq in chance_freq.items()}
        
        # Étape 4: Extraction des séquences historiques pour TimesFM
        progress.set_step(4, "Préparation des séquences pour TimesFM")
        historical_sequences = {}
        
        # Extraire les séquences pour chaque position
        for col in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'numero_chance']:
            historical_sequences[col] = self.df[col].tolist()
        
        # Statistiques complètes
        stats = {
            "frequencies": {
                "boules": dict(boules_freq),
                "chance": dict(chance_freq)
            },
            "probabilities": {
                "boules": boules_prob,
                "chance": chance_prob
            },
            "historical_sequences": historical_sequences,
            "metadata": {
                "total_draws": len(self.df),
                "total_boules_count": total_boules,
                "total_chances_count": total_chances,
                "date_range": {
                    "first": self.df.iloc[0]['date_de_tirage'],
                    "last": self.df.iloc[-1]['date_de_tirage']
                }
            }
        }
        
        # Finaliser la barre de progression
        progress.finish("Statistiques calculées")
        
        self.stats_data = stats
        return stats
    
    def save_stats(self) -> bool:
        """Sauvegarde les statistiques calculées"""
        if self.stats_data is None:
            self.stats_data = self.calculate_frequencies()
        
        if self.stats_data:
            return self.storage.save_stats(self.stats_data)
        return False
    
    def load_stats(self) -> Dict[str, Any]:
        """Charge les statistiques depuis le stockage"""
        stats = self.storage.load_stats()
        if stats:
            self.stats_data = stats
        return stats or {}
    
    def get_most_frequent(self, top_n: int = 10) -> Dict[str, List[Tuple[int, int]]]:
        """Retourne les chiffres les plus fréquents"""
        if self.stats_data is None:
            self.load_stats()
        
        if not self.stats_data:
            return {"boules": [], "chance": []}
        
        boules_freq = self.stats_data["frequencies"]["boules"]
        chance_freq = self.stats_data["frequencies"]["chance"]
        
        boules_sorted = sorted(boules_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        chance_sorted = sorted(chance_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "boules": boules_sorted,
            "chance": chance_sorted
        }
    
    def get_least_frequent(self, bottom_n: int = 10) -> Dict[str, List[Tuple[int, int]]]:
        """Retourne les chiffres les moins fréquents"""
        if self.stats_data is None:
            self.load_stats()
        
        if not self.stats_data:
            return {"boules": [], "chance": []}
        
        boules_freq = self.stats_data["frequencies"]["boules"]
        chance_freq = self.stats_data["frequencies"]["chance"]
        
        boules_sorted = sorted(boules_freq.items(), key=lambda x: x[1])[:bottom_n]
        chance_sorted = sorted(chance_freq.items(), key=lambda x: x[1])[:10]
        
        return {
            "boules": boules_sorted,
            "chance": chance_sorted
        }
    
    def get_probability_weights(self) -> Dict[str, Dict[int, float]]:
        """Retourne les poids de probabilité pour pondération"""
        if self.stats_data is None:
            self.load_stats()
        
        if not self.stats_data:
            return {"boules": {}, "chance": {}}
        
        return self.stats_data.get("probabilities", {"boules": {}, "chance": {}})
    
    def display_summary(self):
        """Affiche un résumé des statistiques"""
        if self.stats_data is None:
            self.load_stats()
        
        if not self.stats_data:
            print("❌ Aucune statistique disponible")
            return
        
        meta = self.stats_data["metadata"]
        most_freq = self.get_most_frequent(5)
        least_freq = self.get_least_frequent(5)
        
        print(f"\n📊 STATISTIQUES LOTO ({meta['total_draws']} tirages)")
        print(f"📅 Période: {meta['date_range']['first']} → {meta['date_range']['last']}")
        
        print(f"\n🔥 Boules les plus fréquentes:")
        for num, freq in most_freq["boules"]:
            prob = self.stats_data["probabilities"]["boules"][num] * 100
            print(f"   {num:2d}: {freq:4d} fois ({prob:.1f}%)")
        
        print(f"\n❄️  Boules les moins fréquentes:")
        for num, freq in least_freq["boules"]:
            prob = self.stats_data["probabilities"]["boules"][num] * 100
            print(f"   {num:2d}: {freq:4d} fois ({prob:.1f}%)")
        
        print(f"\n🍀 Numéros Chance les plus fréquents:")
        for num, freq in most_freq["chance"]:
            prob = self.stats_data["probabilities"]["chance"][num] * 100
            print(f"   {num:2d}: {freq:4d} fois ({prob:.1f}%)")