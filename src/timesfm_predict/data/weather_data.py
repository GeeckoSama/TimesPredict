"""
Processeur de données météorologiques pour TimesFM
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()


class WeatherDataProcessor:
    """Classe pour récupérer et traiter les données météorologiques"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Clé API OpenWeatherMap (si None, cherche dans .env)
        """
        self.api_key = api_key or os.getenv('WEATHER_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.data = None
        
    def fetch_current_weather(self, city: str, country_code: str = "") -> Dict:
        """
        Récupère les données météo actuelles
        
        Args:
            city: Nom de la ville
            country_code: Code pays (optionnel, ex: "FR")
            
        Returns:
            Dictionnaire avec données météo
        """
        if not self.api_key:
            raise ValueError("Clé API manquante. Définissez WEATHER_API_KEY dans .env")
            
        location = f"{city},{country_code}" if country_code else city
        
        url = f"{self.base_url}/weather"
        params = {
            "q": location,
            "appid": self.api_key,
            "units": "metric",
            "lang": "fr"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "weather_main": data["weather"][0]["main"],
                "weather_description": data["weather"][0]["description"],
                "wind_speed": data.get("wind", {}).get("speed", 0),
                "cloudiness": data["clouds"]["all"],
                "city": data["name"],
                "country": data["sys"]["country"]
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Erreur API: {e}")
            raise
        except KeyError as e:
            print(f"Format de réponse inattendu: {e}")
            raise
            
    def fetch_historical_weather(self, 
                                lat: float, 
                                lon: float, 
                                start_date: datetime,
                                end_date: datetime) -> pd.DataFrame:
        """
        Récupère les données météo historiques
        Note: Nécessite un abonnement payant pour OpenWeatherMap
        
        Args:
            lat: Latitude
            lon: Longitude  
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            DataFrame avec données historiques
        """
        if not self.api_key:
            raise ValueError("Clé API manquante")
            
        # Cette fonction nécessite l'API "One Call API 3.0" payante
        print("Note: Les données historiques nécessitent un abonnement payant OpenWeatherMap")
        print("Génération de données d'exemple à la place...")
        
        return self.generate_sample_weather_data(start_date, end_date)
        
    def generate_sample_weather_data(self, 
                                   start_date: datetime,
                                   end_date: datetime,
                                   location_name: str = "Ville Exemple") -> pd.DataFrame:
        """
        Génère des données météo d'exemple pour les tests
        
        Args:
            start_date: Date de début
            end_date: Date de fin
            location_name: Nom de la localisation
            
        Returns:
            DataFrame avec données météo synthétiques
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Température avec saisonnalité
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temp_noise = np.random.normal(0, 3, n_days)
        temperature = base_temp + temp_noise
        
        # Humidité
        humidity = np.random.normal(60, 15, n_days)
        humidity = np.clip(humidity, 20, 95)
        
        # Pression atmosphérique
        pressure = np.random.normal(1013, 20, n_days)
        
        # Vitesse du vent
        wind_speed = np.random.exponential(3, n_days)
        wind_speed = np.clip(wind_speed, 0, 25)
        
        # Couverture nuageuse
        cloudiness = np.random.beta(2, 3, n_days) * 100
        
        # Précipitations (corrélées à la couverture nuageuse)
        rain_prob = cloudiness / 100
        precipitation = np.where(
            np.random.random(n_days) < rain_prob * 0.3,
            np.random.exponential(5),
            0
        )
        
        weather_data = pd.DataFrame({
            'date': dates,
            'temperature': temperature.round(1),
            'humidity': humidity.round(0),
            'pressure': pressure.round(1),
            'wind_speed': wind_speed.round(1),
            'cloudiness': cloudiness.round(0),
            'precipitation': precipitation.round(1),
            'location': location_name
        })
        
        self.data = weather_data
        print(f"Données météo d'exemple générées: {len(weather_data)} jours")
        
        return weather_data
        
    def load_csv(self, 
                 filepath: str,
                 date_column: str = "date",
                 **kwargs) -> pd.DataFrame:
        """
        Charge les données météo depuis un CSV
        
        Args:
            filepath: Chemin vers le fichier CSV
            date_column: Nom de la colonne de date
            **kwargs: Arguments pour pd.read_csv
            
        Returns:
            DataFrame avec données météo
        """
        try:
            self.data = pd.read_csv(filepath, **kwargs)
            
            if date_column in self.data.columns:
                self.data[date_column] = pd.to_datetime(self.data[date_column])
                self.data = self.data.sort_values(date_column)
                
            print(f"Données météo chargées: {len(self.data)} lignes")
            return self.data
            
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            raise
            
    def prepare_covariates(self, 
                          weather_columns: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Prépare les covariables météo pour TimesFM
        
        Args:
            weather_columns: Colonnes à utiliser (si None, utilise toutes les colonnes numériques)
            
        Returns:
            Dictionnaire avec les covariables
        """
        if self.data is None:
            raise ValueError("Aucune donnée météo chargée")
            
        # Sélection des colonnes
        if weather_columns is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            weather_columns = [col for col in numeric_cols if col != 'date']
            
        covariates = {}
        for col in weather_columns:
            if col in self.data.columns:
                covariates[col] = self.data[col].values
            else:
                print(f"Avertissement: Colonne '{col}' non trouvée")
                
        print(f"Covariables préparées: {list(covariates.keys())}")
        return covariates
        
    def create_weather_features(self) -> pd.DataFrame:
        """
        Crée des caractéristiques météo dérivées
        
        Returns:
            DataFrame avec caractéristiques enrichies
        """
        if self.data is None:
            raise ValueError("Aucune donnée chargée")
            
        enhanced = self.data.copy()
        
        # Indice de confort (température + humidité)
        if 'temperature' in enhanced.columns and 'humidity' in enhanced.columns:
            enhanced['comfort_index'] = (
                enhanced['temperature'] * (1 - enhanced['humidity'] / 100)
            )
            
        # Catégories de température
        if 'temperature' in enhanced.columns:
            enhanced['temp_category'] = pd.cut(
                enhanced['temperature'],
                bins=[-50, 0, 10, 20, 30, 50],
                labels=['très_froid', 'froid', 'frais', 'agréable', 'chaud']
            )
            
        # Pluie binaire
        if 'precipitation' in enhanced.columns:
            enhanced['is_rainy'] = (enhanced['precipitation'] > 0).astype(int)
            
        # Vent fort
        if 'wind_speed' in enhanced.columns:
            enhanced['strong_wind'] = (enhanced['wind_speed'] > 10).astype(int)
            
        print(f"Caractéristiques météo enrichies: {enhanced.shape[1]} colonnes")
        return enhanced
        
    def get_summary(self) -> Dict:
        """Retourne un résumé des données météo"""
        if self.data is None:
            return {"status": "Aucune donnée chargée"}
            
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        summary = {
            "rows": len(self.data),
            "columns": list(self.data.columns),
            "numeric_columns": list(numeric_cols)
        }
        
        if 'date' in self.data.columns:
            summary["date_range"] = (
                str(self.data['date'].min()),
                str(self.data['date'].max())
            )
            
        # Statistiques des colonnes numériques
        for col in numeric_cols:
            summary[f"{col}_stats"] = {
                "mean": float(self.data[col].mean()),
                "std": float(self.data[col].std()),
                "min": float(self.data[col].min()),
                "max": float(self.data[col].max())
            }
            
        return summary