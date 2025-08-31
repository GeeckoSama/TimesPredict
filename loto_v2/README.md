# Loto V2 - Prédicteur Simplifié

## 🎯 Description

Version 2 simplifiée du prédicteur loto utilisant une approche statistique pondérée + fine-tuning TimesFM. Architecture modulaire minimaliste remplaçant le système complexe de la v1.

## 🏗️ Architecture

```
loto_v2/
├── loto_v2.py              # CLI principal
├── modules/
│   ├── storage.py         # Persistance JSON/pickle  
│   ├── stats.py          # Analyse fréquences historiques
│   ├── finetuning.py     # Fine-tuning TimesFM
│   ├── prediction.py     # Prédictions pondérées
│   └── validation.py     # Séries de prédictions
└── data_v2/              # Données générées
    ├── stats/           # Statistiques (.json)
    └── models/          # Modèles fine-tunés (.pkl)
```

## 🚀 Utilisation

```bash
# Lancer le CLI
python3 loto_v2.py

# Menu interactif avec 6 options :
# 1. Calculer statistiques historiques
# 2. Fine-tuner le modèle TimesFM  
# 3. Générer 1 prédiction
# 4. Série de N prédictions
# 5. Afficher statut détaillé
# 6. Quitter
```

## 📊 Workflow Recommandé

1. **Calculer les statistiques** (option 1) - Analyse historique des 5616 tirages
2. **Fine-tuner TimesFM** (option 2) - Adaptation sur données loto (optionnel)
3. **Générer prédictions** (option 3/4) - Combinaisons pondérées

## 🔧 Fonctionnalités V2

### ✅ Nouveautés vs V1
- **CLI minimaliste** (1 fichier vs architecture complexe)
- **5 modules simples** (vs 15+ fichiers)
- **Output non-verbeux** (vs logs détaillés)
- **Opérations indépendantes** avec statut persistant
- **Pondération par fréquences** historiques (49 ans de données)
- **Fine-tuning intégré** (vs modèles multiples)

### 📈 Méthodes de Prédiction
- **Statistique pure** : Pondération par fréquences historiques
- **TimesFM pur** : Modèle de base (simulation)
- **Hybride** : TimesFM fine-tuné + pondération statistique

### 🎲 Analyse de Séries
- Génération de 10-1000 prédictions
- Analyse des fréquences d'apparition
- Combinaison optimale par consensus
- Comparaison avec données historiques
- Métriques de confiance

## 📋 États du Système

Le CLI affiche automatiquement :
- **Statistiques** : ✅ calculées / ❌ à faire
- **Fine-tuning** : ✅ fait / ❌ à faire

## 🛠️ Dépendances

- Python 3.10+
- pandas, numpy
- timesfm (Google Research)
- Collections (standard lib)

## 🎨 Interface Moderne

### Barres de Progression
- **ProgressBar** : Barres détaillées avec ETA
- **Loading Animation** : Spinners pour tâches courtes  
- **Simple Progress** : Progress inline minimaliste

```bash
# Voir les barres en action
python3 demo_progress.py
```

## 💡 Philosophie V2

- **Simple avant tout** - CLI direct sans menus complexes
- **Modulaire** - Chaque fonction isolée et testable
- **Statistiques first** - Données historiques comme base
- **TimesFM optionnel** - Fonctionne sans fine-tuning
- **Output minimal** - Informations essentielles uniquement

## 🎯 Cas d'Usage

- **Recherche** : Analyse des patterns historiques
- **Prédiction ponctuelle** : 1 combinaison rapide
- **Analyse de masse** : Tendances sur N prédictions
- **Validation** : Comparaison méthodes statistiques vs ML