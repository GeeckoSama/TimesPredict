"""
TimesPredict - Prédiction de séries temporelles avec TimesFM
Spécialisé dans la prédiction des ventes avec données météorologiques
"""

__version__ = "0.1.0"
__author__ = "TimesPredict Project"

from .models.timesfm_wrapper import TimesFMPredictor
from .data.sales_data import SalesDataProcessor
from .data.weather_data import WeatherDataProcessor

__all__ = [
    "TimesFMPredictor",
    "SalesDataProcessor", 
    "WeatherDataProcessor"
]