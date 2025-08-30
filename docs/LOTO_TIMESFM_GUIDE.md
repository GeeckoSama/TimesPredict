# 🎰 Guide Complet - Prédicteur Loto TimesFM

## ⚠️ AVERTISSEMENT IMPORTANT

**Usage éducatif et de recherche uniquement**
- Ce logiciel ne garantit **aucun gain** au loto
- Le loto est un jeu de **hasard pur**
- Les prédictions sont basées sur l'analyse de patterns statistiques
- **Jouez avec modération** et de manière responsable

## 🚀 Installation et Lancement

### Prérequis
- Python 3.10, 3.11 ou 3.12
- Fichier `data/raw/loto_201911.csv` (vos 910 tirages historiques)
- Modules TimesFM installés

### Lancement
```bash
python loto_timesfm_cli.py
```

## 🎯 Fonctionnalités Principales

### 1. 🔮 Génération de Prédictions
- **6 modèles TimesFM coordonnés** (1 par boule + 1 pour le numéro chance)
- **Optimisation post-TimesFM** avec 5 stratégies différentes
- **1-20 combinaisons** générées selon vos préférences
- **Scores de confiance** pour chaque prédiction

### 2. 📊 Analyse Statistique Complète
- **Fréquences de sortie** (numéros chauds/froids)
- **Patterns temporels** (jours de la semaine, mois, années)
- **Tests de randomité** (Kolmogorov-Smirnov, runs, autocorrélations)
- **Corrélations** entre boules et avec le temps
- **Détection de cycles** (analyse spectrale FFT)
- **Analyse des séquences** (gaps, répétitions)

### 3. 🧪 Système de Validation (Backtest)
- **Test sur historique** (50-110 derniers tirages)
- **Métriques de performance** détaillées
- **Significativité statistique** vs hasard
- **Recommandations d'usage** basées sur les performances

### 4. 🎯 Stratégies de Génération

#### **TimesFM Direct**
- Prédictions brutes des 6 modèles TimesFM
- Variantes avec ajustements mineurs
- Score basé sur la confiance TimesFM

#### **Statistical Weighted**
- Basé sur les fréquences historiques inverses
- Favorise les numéros moins sortis récemment
- Pondération selon l'analyse statistique

#### **Hybrid Optimized**
- Combine TimesFM + insights statistiques
- 2-3 boules de TimesFM + complétion statistique
- Score hybride pondéré

#### **Frequency Balanced**
- Équilibrage chaud/froid/modéré
- 1 chaud + 1 froid + 3 modérés
- Évite les extrêmes fréquentiels

#### **Pattern Aware**
- Évite les patterns trop évidents
- Maximum 2 numéros consécutifs
- Équilibrage parité (pairs/impairs)

## 📋 Interface CLI

### Menu Principal
1. **Prédictions uniquement** - Génération rapide
2. **Analyse statistique** - Exploration des données  
3. **Backtest** - Validation des performances
4. **Analyse + Prédictions** - Processus complet
5. **Quitter**

### Configuration
- **Nombre de combinaisons** : 1-20 (défaut: 5)
- **Backend** : CPU (stable) ou GPU (rapide si CUDA)
- **Modèle TimesFM** : 
  - `200M v1.0 PyTorch` (rapide)
  - `500M v2.0 PyTorch` (précis, **recommandé**)
- **Mode** : TimesFM réel ou simulation (tests)

## 🔧 Architecture Technique

### Structure des Modules
```
src/loto_predict/
├── data/loto_data_processor.py      # Traitement données françaises
├── models/multi_timesfm_predictor.py # 6 modèles TimesFM coordonnés
├── analysis/loto_stat_analyzer.py   # Analyse statistique avancée
├── optimization/combination_generator.py # 5 stratégies d'optimisation
└── validation/backtest_validator.py # Validation et métriques
```

### Pipeline de Prédiction
1. **Chargement** des 910 tirages historiques
2. **Création** de 6 séries temporelles (boule_1...5 + chance)
3. **Prédiction** avec 6 modèles TimesFM séparés
4. **Post-traitement** (correction doublons, contraintes)
5. **Génération** de combinaisons via 5 stratégies
6. **Scoring** et classement des meilleures combinaisons

## 📊 Métriques et Évaluation

### Métriques de Précision
- **Boules exactes** : 0-5 (nombre de boules correctement prédites)
- **Numéro chance exact** : 0-1 (chance correcte)
- **Score total** : Pondéré 80% boules + 20% chance
- **Proximité** : Distance minimale aux numéros réels

### Métriques Loto Spécifiques
- **Simulation des rangs de gain** (1-9 selon règles officielles)
- **Analyse des sommes** prédites vs réelles
- **Patterns de parité** (pairs/impairs)

### Tests Statistiques
- **Kolmogorov-Smirnov** : Distribution uniforme
- **Chi-carré** : Uniformité des fréquences
- **Tests de runs** : Randomité des séquences
- **T-test** : Amélioration vs hasard

## 🎯 Utilisation Optimale

### Première Utilisation
1. **Démarrer** par l'analyse statistique complète
2. **Examiner** les patterns et biais détectés
3. **Lancer** un backtest pour évaluer les performances
4. **Générer** des prédictions selon les recommandations

### Interprétation des Résultats

#### Scores de Confiance
- **0.7-1.0** : Prédiction cohérente avec les patterns
- **0.5-0.7** : Prédiction modérément fiable
- **0.0-0.5** : Prédiction incertaine

#### Stratégies Recommandées
- **Hybrid Optimized** : Généralement le meilleur équilibre
- **TimesFM Direct** : Si le modèle montre de bonnes performances
- **Frequency Balanced** : Pour un approche conservative

### Bonnes Pratiques
- **Comparez** plusieurs stratégies
- **Analysez** les résultats du backtest
- **Ne jamais** considérer comme garanties de gain
- **Utilisez** pour comprendre les patterns statistiques

## 🔍 Analyse des Patterns Détectés

Votre fichier `loto_201911.csv` contient **910 tirages** - excellent pour l'analyse !

### Patterns Typiques Analysés
- **Fréquences** : Numéros sortis plus/moins souvent
- **Cycles temporels** : Variations selon jours/mois/années  
- **Séquences** : Intervalles entre sorties d'un même numéro
- **Corrélations** : Relations entre positions de boules
- **Parité** : Distribution pairs/impairs

## 💾 Fichiers Générés

### Prédictions
- `loto_predictions_YYYYMMDD_HHMM.txt` - Combinaisons générées
- Format lisible avec scores et méthodes

### Analyses  
- `loto_analysis_YYYYMMDD_HHMM.json` - Résultats statistiques complets
- Données JSON pour analyse ultérieure

### Backtest
- `loto_backtest_YYYYMMDD_HHMM.json` - Résultats de validation
- Métriques de performance détaillées

## 🤔 Questions Fréquentes

### **Q: Les prédictions sont-elles fiables ?**
**R**: Les prédictions analysent des patterns statistiques mais ne peuvent pas prédire le hasard. Utilisez pour l'éducation et la recherche uniquement.

### **Q: Quelle stratégie choisir ?**
**R**: Commencez par \"Hybrid Optimized\" qui combine TimesFM et statistiques. Ajustez selon les résultats du backtest.

### **Q: Le backtest montre de mauvaises performances**
**R**: Normal ! Le loto est aléatoire. Si les performances sont proches du hasard, c'est attendu. Cherchez des améliorations marginales.

### **Q: Combien de combinaisons générer ?**
**R**: 5-10 combinaisons offrent une bonne diversité. Plus peut diluer la qualité.

### **Q: GPU ou CPU ?**
**R**: CPU est plus stable. GPU uniquement si vous avez CUDA installé et configuré.

## ⚡ Performance et Optimisation

### Temps d'Exécution Typiques
- **Analyse statistique** : 1-2 minutes
- **Chargement TimesFM** : 2-3 minutes (première fois)
- **Génération prédictions** : 1-2 minutes
- **Backtest complet** : 15-30 minutes

### Optimisations Possibles
- **Mode simulation** pour tests rapides
- **Moins de tirages** de backtest pour vitesse
- **CPU multicœur** pour parallélisation

## 🔬 Aspects Scientifiques

### Hypothèse de Recherche
Explorer si des **modèles de séries temporelles avancés** (TimesFM) peuvent détecter des **micro-patterns** ou **biais subtils** dans les systèmes de tirage, même théoriquement aléatoires.

### Méthodologie
1. **Multi-modèles** : 6 TimesFM spécialisés par composant
2. **Post-traitement** : Optimisation combinatoire
3. **Validation rigoureuse** : Backtest avec métriques statistiques
4. **Comparaison** : Performance vs hasard pur

### Limitations Connues
- **Données limitées** : 910 tirages vs millions nécessaires
- **Randomité intrinsèque** : Loto conçu pour être imprévisible
- **Overfitting** possible sur patterns temporaires
- **Pas de causalité** : Corrélations ≠ prédictibilité

## 📈 Conclusion

Ce système représente une **expérimentation avancée** d'application de l'IA (TimesFM) à un problème de **randomité pure**. 

Il offre :
- ✅ **Analyses statistiques** approfondies
- ✅ **Méthodologie scientifique** rigoureuse  
- ✅ **Validation empirique** via backtest
- ✅ **Transparence** des limitations

**Rappel** : Usage éducatif uniquement. Les gains au loto restent dus au hasard, pas aux prédictions algorithmiques.

---

*Bon courage dans vos expérimentations !* 🎰📊🤖