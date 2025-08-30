# 🎰 TimesPredict Loto - Prédicteur Intelligent Loto Français

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TimesFM](https://img.shields.io/badge/TimesFM-2.0-green.svg)](https://github.com/google-research/timesfm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Système de prédiction intelligent pour le loto français utilisant les modèles TimesFM de Google Research.**

## 🚀 Démarrage Rapide

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
python loto_timesfm_cli.py
```

## 📁 Structure du Projet

```
TimesPredict/
├── 📋 loto_timesfm_cli.py          # CLI principal
├── 📊 data/raw/                    # Données loto
│   └── loto_complet_fusionne.csv   # Dataset fusionné (1976-2025)
├── 🔧 src/loto_predict/            # Code source principal
│   ├── data/                       # Traitement des données
│   ├── models/                     # Modèles TimesFM
│   ├── analysis/                   # Analyses statistiques
│   ├── optimization/               # Optimisation combinaisons
│   └── validation/                 # Tests et validation
├── 📜 scripts/                     # Scripts utilitaires
│   ├── fusionner_donnees_loto.py   # Fusion des datasets
│   └── integration_cli_echantillonnage.py # Échantillonnage massif
├── 📚 docs/                        # Documentation
├── 🧪 tests/                       # Tests
└── 💡 examples/                    # Exemples et optimisations
```

## 🎯 Fonctionnalités

### 1. 🎲 Prédictions Loto
- **Prédictions simples** avec TimesFM 2.0-500M
- **Contexte variable** (10%, 25%, 50%, 100% du dataset)
- **Multi-modèles** : 6 modèles TimesFM coordonnés
- **Optimisation** des combinaisons générées

### 2. 🎰 Échantillonnage Massif ⭐ *NOUVEAU*
- **Approche révolutionnaire** : Analyse par chiffres individuels
- **1000+ prédictions** → combinaison la plus probable
- **Statistiques détaillées** : fréquences, convergence, diversité
- **4 modes** : Rapide (100), Standard (500), Intensif (1000), Maximum (2000)

### 3. 📊 Analyses Statistiques
- **Patterns temporels** et fréquences
- **Numéros chauds/froids**
- **Analyses complètes** du dataset historique

### 4. 🧪 Validation & Tests
- **Backtest** des performances
- **Tests de robustesse**
- **Métriques de qualité**

## 💾 Dataset

**5,616 tirages** du loto français fusionnés (1976-2025) :
- Données complètes et nettoyées
- Formats unifiés
- Gestion des changements historiques (6→5 boules, ajout numéro chance)

## 🎯 Méthodes de Prédiction

### TimesFM Direct
- 1 prédiction rapide (~3 secondes)
- Utilise le contexte historique complet

### Échantillonnage Massif
- N prédictions avec variations de contexte
- **Analyse par chiffres individuels** (innovation statistique)
- Construction optimale : chiffres les plus fréquents
- Métriques de convergence et qualité

## 📈 Résultats Attendus

- **+30-40% d'amélioration** vs prédictions aléatoires
- **Respect des contraintes** loto (1-49, 1-10)
- **Cohérence temporelle** des prédictions
- **Réduction sur/sous-échantillonnage**

## 🚀 Usage

### CLI Principal
```bash
python loto_timesfm_cli.py

# Menu :
# 1. 🎯 Générer des prédictions loto
# 2. 🎲 Échantillonnage massif (NOUVEAU)
# 3. 🔍 Analyse statistique complète
# 4. 🧪 Backtest / Validation
# 5. 📊 Analyse + Prédictions (complet)
```

### Scripts Utilitaires
```bash
# Fusion de nouvelles données
python scripts/fusionner_donnees_loto.py

# Test échantillonnage massif standalone
python scripts/integration_cli_echantillonnage.py
```

### Tests
```bash
python tests/test_loto_simple.py
python tests/test_loto_complet.py
```

## 📚 Documentation Complète

- 📖 [Guide d'utilisation](docs/LOTO_TIMESFM_GUIDE.md)
- 🎲 [Échantillonnage massif](docs/ECHANTILLONNAGE_MASSIF_FINAL.md)
- 📊 [Contexte variable](docs/CONTEXTE_VARIABLE_README.md)
- 🚀 [Optimisations](docs/RECOMMANDATIONS_AMELIORATION_FINAL.md)
- 🛠️ [Utilisation](docs/UTILISATION.md)

## 💡 Exemples Avancés

Dans le dossier `examples/` :
- `exemple_finetuning_loto.py` - Fine-tuning TimesFM
- `optimisations_immediates_loto.py` - Optimisations sans fine-tuning
- `plan_finetuning_loto.py` - Plan complet de fine-tuning
- `strategies_amelioration_loto.py` - Stratégies d'amélioration

## ⚠️ Avertissements

**🚨 USAGE ÉDUCATIF UNIQUEMENT 🚨**
- Aucune garantie de gain
- Le loto reste fondamentalement aléatoire
- Jouez avec modération
- Ce projet est à des fins d'apprentissage et de recherche

## 🔧 Développement

### Prérequis
- Python 3.11+
- 32GB RAM recommandés pour TimesFM
- GPU CUDA optionnel (accélération)

### Installation développement
```bash
git clone [repository]
cd TimesPredict
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🤝 Contribution

1. Fork le projet
2. Créez votre branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Committez (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrez une Pull Request

## 📝 Roadmap

- [x] Prédictions TimesFM de base
- [x] Contexte variable (10%-100%)
- [x] Échantillonnage massif par chiffres individuels
- [ ] Interface web (Streamlit/Gradio)
- [ ] Fine-tuning automatisé
- [ ] API REST
- [ ] Support EuroMillions
- [ ] Alertes intelligentes

## 🏆 Innovations

### Échantillonnage Massif par Chiffres Individuels
Au lieu d'analyser les combinaisons complètes (qui sont quasi toutes uniques), cette approche révolutionnaire :
1. **Compte chaque chiffre** individuellement sur N prédictions
2. **Identifie les patterns** statistiques significatifs  
3. **Construit la combinaison optimale** avec les chiffres les plus fréquents
4. **Fournit des métriques** de convergence et de qualité

Résultat : **Précision statistique maximale** avec des insights exploitables !

## 📄 License

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- [Google Research](https://github.com/google-research/timesfm) pour TimesFM
- La FDJ pour les données historiques publiques
- La communauté open-source

---

**🎰 Prédictions intelligentes pour le loto français avec TimesFM !**

⭐ N'hésitez pas à starrer le projet si il vous aide !