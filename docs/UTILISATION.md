# 🔮 TimesFM CLI - Guide d'Utilisation Rapide

## Lancement rapide

```bash
python timesfm_cli.py
```

## Que fait le script ?

Le script vous guide étape par étape :

1. **📂 Sélection du fichier** - Pointez vers votre CSV de données
2. **🔍 Détection automatique** - Le format CSV est détecté automatiquement  
3. **🎯 Choix de la colonne** - Sélectionnez quelle colonne prédire
4. **⚙️ Configuration** - Paramétrez horizon, modèle, backend
5. **🚀 Prédiction** - TimesFM génère les prédictions
6. **📊 Résultats** - Affichage détaillé + sauvegarde optionnelle

## Formats de données supportés

### CSV Simple
```
date,ventes
2024-01-01,120.50
2024-01-02,135.20
```

### CSV Français (séparateur ; et virgule décimale)
```
date;ventes
01/01/2024;120,50
02/01/2024;135,20
```

### CSV Multi-colonnes
```
date,produit_A,produit_B,produit_C
2024-01-01,120,80,200
2024-01-02,135,90,210
```

## Exemple d'utilisation

```
👉 Chemin vers votre fichier de données: data/raw/commerce_07_2025.csv
✅ Fichier trouvé: data/raw/commerce_07_2025.csv

🔍 ANALYSE DU FICHIER: commerce_07_2025.csv
✅ Format détecté - Séparateur: ';', Encodage: utf-8

🎯 SÉLECTION DE LA COLONNE À PRÉDIRE:
Colonnes disponibles:
   1. date (ex: ['01/07/2024' '02/07/2024' '03/07/2024'])
   2. ventes (ex: ['786,35' '520,85' '928,90'])

👉 Numéro de la colonne à prédire (1-2): 2
✅ Colonne sélectionnée: ventes

⚙️ CONFIGURATION DE LA PRÉDICTION:
👉 Nombre de périodes à prédire (défaut: 30): 15
👉 Choix (1-2, défaut: 1): 1
👉 Choix du modèle (1-3, défaut: 3): 3
👉 Choix (1-2, défaut: 1): 1

🚀 EXÉCUTION DE LA PRÉDICTION:
   Initialisation du modèle TimesFM...
   Chargement du modèle (peut prendre 1-2 minutes)...
   Génération des prédictions...

🎯 RÉSULTATS DES PRÉDICTIONS
📊 PRÉDICTIONS DÉTAILLÉES:
   Période + 1:    1245.67
   Période + 2:    1189.23
   [...]
```

## Paramètres configurables

- **Horizon** : Nombre de périodes à prédire (1-365)
- **Backend** : CPU (recommandé) ou GPU (si CUDA disponible)
- **Modèle TimesFM** :
  - 200M v1.0 - Plus rapide, moins précis (20 couches)
  - 200M v1.0 PyTorch - Stable et rapide (20 couches)
  - **500M v2.0 PyTorch** - Plus précis, contexte 4x plus long (50 couches) 🌟 RECOMMANDÉ
- **Mode** : TimesFM réel ou simulation (pour tests)

## Fichiers générés

Sauvegarde optionnelle au format :
```
predictions_[nom_fichier]_[YYYYMMDD_HHMM].csv
```

## Conseils

- **Données minimum** : 20-30 points historiques recommandés
- **Format préféré** : CSV avec colonnes date + valeurs numériques
- **Mode simulation** : Idéal pour tester rapidement sans charger TimesFM
- **Première utilisation** : Testez d'abord en mode simulation