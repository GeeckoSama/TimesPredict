# TimesPredict 🚀

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TimesFM](https://img.shields.io/badge/TimesFM-2.0-green.svg)](https://github.com/google-research/timesfm)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Projet de prédiction de séries temporelles utilisant TimesFM de Google Research, spécialisé dans la prédiction des ventes avec intégration des données météorologiques.**

## ✨ Fonctionnalités

- 🏪 **Prédiction des ventes** : Modèle TimesFM optimisé pour les données commerciales
- 🌤️ **Covariables météorologiques** : Intégration des données météo pour améliorer les prédictions
- 📊 **Données financières** : Support complet pour les séries temporelles de vente
- 🎯 **Interface simplifiée** : API facile à utiliser pour tests et expérimentations
- 📈 **Visualisations automatiques** : Graphiques générés automatiquement
- 🔌 **API météo** : Connexion OpenWeatherMap pour données temps réel

## Installation

1. Créer un environnement virtuel Python :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Structure du projet

```
TimesPredict/
├── src/timesfm_predict/     # Code source principal
│   ├── data/               # Modules de gestion des données
│   ├── models/             # Wrappers et configurations de modèles
│   ├── utils/              # Utilitaires divers
│   └── examples/           # Scripts d'exemple
├── data/                   # Données
│   ├── raw/               # Données brutes
│   └── processed/         # Données prétraitées
├── notebooks/             # Jupyter notebooks pour exploration
└── tests/                 # Tests unitaires
```

## 🚀 Usage rapide

### Installation automatique
```bash
python install_and_test.py
```

### Exemple basique
```python
from timesfm_predict.models.timesfm_wrapper import TimesFMPredictor
from timesfm_predict.data.sales_data import SalesDataProcessor

# Créer des données d'exemple
processor = SalesDataProcessor()
sales_data = processor.create_sample_data(periods=365)

# Préparer pour TimesFM
sales_array, metadata = processor.prepare_for_timesfm()

# Prédiction
predictor = TimesFMPredictor(horizon_len=30)
predictor.load_model()
results = predictor.predict_sales(sales_array)
```

### Exemples complets
- **Base** : `python src/timesfm_predict/examples/basic_sales_prediction.py`
- **Avec météo** : `python src/timesfm_predict/examples/sales_with_weather.py`

## 📚 Documentation

### API principale

#### TimesFMPredictor
```python
predictor = TimesFMPredictor(
    horizon_len=30,      # Nombre de jours à prédire
    backend="gpu"        # "gpu" ou "cpu"
)
```

#### SalesDataProcessor
```python
processor = SalesDataProcessor()

# Charger des données CSV
data = processor.load_csv("sales.csv", date_column="date", sales_column="revenue")

# Ou créer des données d'exemple
data = processor.create_sample_data(periods=365, base_sales=1000)
```

#### WeatherDataProcessor
```python
weather = WeatherDataProcessor(api_key="your_key")

# Générer des données d'exemple
weather_data = weather.generate_sample_weather_data(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)
```

## 🔧 Configuration

### Variables d'environnement
Copiez `.env.example` vers `.env` :
```bash
# API météo (optionnel)
WEATHER_API_KEY=your_openweathermap_api_key

# Chemins
MODEL_CACHE_DIR=./models_cache
DATA_PATH=./data
```

## 📊 Exemple de résultats

Le projet génère automatiquement des visualisations :
- Graphiques des prédictions vs données historiques
- Analyse de corrélation météo-ventes
- Intervalles de confiance (expérimental)

## 🤝 Contribution

1. Fork le projet
2. Créez votre branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Committez (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrez une Pull Request

## 📝 Roadmap

- [ ] Interface graphique web (Streamlit/Gradio)
- [ ] Support d'autres APIs météo
- [ ] Modèles ensemble avec TimesFM
- [ ] Export vers formats business (Excel, PowerBI)
- [ ] Alertes automatiques sur seuils
- [ ] API REST pour intégration

## ⚠️ Limitations

- TimesFM nécessite Python 3.11+ et beaucoup de RAM (32GB recommandés)
- L'intégration complète des covariables est expérimentale dans TimesFM
- L'API météo gratuite a des limitations de requêtes

## 📄 License

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- [Google Research](https://github.com/google-research/timesfm) pour TimesFM
- [OpenWeatherMap](https://openweathermap.org/) pour l'API météo
- La communauté open-source pour les outils utilisés

---

**⭐ N'hésitez pas à starrer le projet si il vous aide !**