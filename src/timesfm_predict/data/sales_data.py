"""
Processeur de données de ventes pour TimesFM
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple, Union
from datetime import datetime, timedelta


class SalesDataProcessor:
    """Classe pour traiter et préparer les données de ventes"""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.date_column = None
        self.sales_column = None
        
    def load_csv(self, 
                 filepath: str,
                 date_column: str = "date",
                 sales_column: str = "sales",
                 **kwargs) -> pd.DataFrame:
        """
        Charge les données de ventes depuis un CSV
        
        Args:
            filepath: Chemin vers le fichier CSV
            date_column: Nom de la colonne de date
            sales_column: Nom de la colonne de ventes
            **kwargs: Arguments pour pd.read_csv
            
        Returns:
            DataFrame pandas avec les données
        """
        try:
            self.data = pd.read_csv(filepath, **kwargs)
            self.date_column = date_column
            self.sales_column = sales_column
            
            # Conversion de la colonne date
            if date_column in self.data.columns:
                self.data[date_column] = pd.to_datetime(self.data[date_column])
                self.data = self.data.sort_values(date_column)
                
            print(f"Données chargées: {len(self.data)} lignes")
            print(f"Période: {self.data[date_column].min()} à {self.data[date_column].max()}")
            
            return self.data
            
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            raise
            
    def create_sample_data(self, 
                          start_date: str = "2023-01-01",
                          periods: int = 365,
                          base_sales: float = 1000.0,
                          seasonality: bool = True,
                          trend: float = 0.1,
                          noise_level: float = 0.2) -> pd.DataFrame:
        """
        Crée des données d'exemple pour tester
        
        Args:
            start_date: Date de début
            periods: Nombre de jours
            base_sales: Ventes de base
            seasonality: Ajouter de la saisonnalité
            trend: Tendance (croissance par jour)
            noise_level: Niveau de bruit (0-1)
            
        Returns:
            DataFrame avec données synthétiques
        """
        dates = pd.date_range(start=start_date, periods=periods, freq='D')
        
        # Tendance linéaire
        trend_component = np.arange(periods) * trend
        
        # Saisonnalité (hebdomadaire + mensuelle)
        seasonal_component = 0
        if seasonality:
            # Saisonnalité hebdomadaire (plus de vente en weekend)
            weekly_season = np.sin(2 * np.pi * np.arange(periods) / 7) * 0.3
            # Saisonnalité mensuelle
            monthly_season = np.sin(2 * np.pi * np.arange(periods) / 30) * 0.2
            seasonal_component = weekly_season + monthly_season
            
        # Bruit aléatoire
        noise = np.random.normal(0, noise_level, periods)
        
        # Combinaison
        sales = base_sales * (1 + trend_component + seasonal_component + noise)
        
        # S'assurer que les ventes sont positives
        sales = np.maximum(sales, base_sales * 0.1)
        
        self.data = pd.DataFrame({
            'date': dates,
            'sales': sales.round(2)
        })
        
        self.date_column = 'date'
        self.sales_column = 'sales'
        
        print(f"Données d'exemple créées: {len(self.data)} jours")
        print(f"Ventes moyennes: {sales.mean():.2f}")
        
        return self.data
        
    def prepare_for_timesfm(self, 
                           context_length: int = 512,
                           target_column: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Prépare les données pour TimesFM
        
        Args:
            context_length: Longueur du contexte historique
            target_column: Colonne cible (si None, utilise sales_column)
            
        Returns:
            Tuple (données_array, métadonnées)
        """
        if self.data is None:
            raise ValueError("Aucune donnée chargée. Appelez load_csv() ou create_sample_data() d'abord.")
            
        target_col = target_column or self.sales_column
        
        if target_col not in self.data.columns:
            raise ValueError(f"Colonne '{target_col}' non trouvée dans les données.")
            
        # Extrait la série temporelle
        sales_series = self.data[target_col].values
        
        # Tronque à la longueur de contexte si nécessaire
        if len(sales_series) > context_length:
            sales_series = sales_series[-context_length:]
            print(f"Données tronquées aux derniers {context_length} points")
            
        # Métadonnées
        metadata = {
            "original_length": len(self.data),
            "context_length": len(sales_series),
            "target_column": target_col,
            "date_range": (
                self.data[self.date_column].iloc[-len(sales_series)],
                self.data[self.date_column].iloc[-1]
            ),
            "statistics": {
                "mean": float(sales_series.mean()),
                "std": float(sales_series.std()),
                "min": float(sales_series.min()),
                "max": float(sales_series.max())
            }
        }
        
        self.processed_data = sales_series
        
        return sales_series, metadata
        
    def add_weather_features(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les caractéristiques météo aux données de ventes
        
        Args:
            weather_data: DataFrame avec données météo
            
        Returns:
            DataFrame combiné
        """
        if self.data is None:
            raise ValueError("Aucune donnée de vente chargée.")
            
        # Fusion sur la date
        combined = self.data.merge(
            weather_data, 
            on=self.date_column, 
            how='left'
        )
        
        print(f"Données combinées: {len(combined)} lignes avec {len(weather_data.columns)-1} variables météo")
        
        return combined
        
    def get_summary(self) -> Dict:
        """Retourne un résumé des données"""
        if self.data is None:
            return {"status": "Aucune donnée chargée"}
            
        return {
            "rows": len(self.data),
            "columns": list(self.data.columns),
            "date_range": (
                str(self.data[self.date_column].min()), 
                str(self.data[self.date_column].max())
            ) if self.date_column else None,
            "sales_stats": {
                "mean": float(self.data[self.sales_column].mean()),
                "std": float(self.data[self.sales_column].std()),
                "min": float(self.data[self.sales_column].min()),
                "max": float(self.data[self.sales_column].max())
            } if self.sales_column else None
        }