#!/usr/bin/env python3
"""
Script CLI principal pour TimesFM - Prédictions temporelles simplifiées
Interface interactive pour paramétrer et lancer des prédictions
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Ajouter le src au path
sys.path.append("src")

from timesfm_predict.models.timesfm_wrapper import TimesFMPredictor


def afficher_titre():
    """Affiche le titre de l'application"""
    print("\n" + "=" * 60)
    print("🔮 TIMESFM CLI - PRÉDICTIONS TEMPORELLES")
    print("=" * 60)
    print("Interface simplifiée pour vos prédictions de séries temporelles")
    print("=" * 60)


def demander_fichier_donnees():
    """Demande à l'utilisateur le fichier de données source"""
    print("\n📂 FICHIER DE DONNÉES:")
    print("Formats supportés: CSV (.csv)")
    
    while True:
        fichier = input("\n👉 Chemin vers votre fichier de données: ").strip()
        
        if not fichier:
            print("❌ Veuillez entrer un chemin de fichier")
            continue
            
        fichier_path = Path(fichier)
        
        if not fichier_path.exists():
            print(f"❌ Fichier non trouvé: {fichier}")
            print("💡 Tip: Utilisez des chemins relatifs comme 'data/raw/mon_fichier.csv'")
            continue
            
        if not fichier.lower().endswith('.csv'):
            print("❌ Seuls les fichiers CSV sont supportés pour le moment")
            continue
            
        print(f"✅ Fichier trouvé: {fichier}")
        return str(fichier_path)


def detecter_format_csv(fichier_path):
    """Détecte automatiquement le format du CSV"""
    print(f"\n🔍 ANALYSE DU FICHIER: {Path(fichier_path).name}")
    
    # Test différents séparateurs et formats
    separateurs = [',', ';', '\t']
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        for sep in separateurs:
            try:
                df = pd.read_csv(fichier_path, sep=sep, encoding=encoding, nrows=5)
                if len(df.columns) >= 2:
                    print(f"✅ Format détecté - Séparateur: '{sep}', Encodage: {encoding}")
                    print(f"   Colonnes: {list(df.columns)}")
                    print(f"   Aperçu: {len(df)} lignes (échantillon)")
                    return sep, encoding
            except:
                continue
    
    # Format par défaut
    print("⚠️  Format non détecté automatiquement, utilisation des paramètres par défaut")
    return ',', 'utf-8'


def charger_donnees(fichier_path):
    """Charge et analyse les données"""
    print("\n📊 CHARGEMENT DES DONNÉES:")
    
    # Détection automatique du format
    sep, encoding = detecter_format_csv(fichier_path)
    
    # Chargement
    try:
        df = pd.read_csv(fichier_path, sep=sep, encoding=encoding)
        print(f"✅ {len(df)} lignes chargées")
        print(f"   Colonnes: {list(df.columns)}")
        
        # Affichage des premières lignes
        print("\n📋 APERÇU DES DONNÉES:")
        print(df.head())
        
        return df, sep, encoding
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None, sep, encoding


def selectionner_colonne_valeurs(df):
    """Permet à l'utilisateur de sélectionner la colonne des valeurs à prédire"""
    print("\n🎯 SÉLECTION DE LA COLONNE À PRÉDIRE:")
    
    colonnes = list(df.columns)
    print("Colonnes disponibles:")
    for i, col in enumerate(colonnes, 1):
        echantillon = df[col].head(3).values
        print(f"   {i}. {col} (ex: {echantillon})")
    
    while True:
        try:
            choix = input(f"\n👉 Numéro de la colonne à prédire (1-{len(colonnes)}): ").strip()
            
            if not choix:
                print("❌ Veuillez entrer un numéro")
                continue
                
            index = int(choix) - 1
            
            if 0 <= index < len(colonnes):
                colonne_choisie = colonnes[index]
                print(f"✅ Colonne sélectionnée: {colonne_choisie}")
                
                # Vérifier que ce sont des valeurs numériques
                try:
                    # Essayer de convertir en numérique
                    valeurs_test = pd.to_numeric(df[colonne_choisie].replace(',', '.', regex=True), errors='coerce')
                    valeurs_valides = valeurs_test.dropna()
                    
                    if len(valeurs_valides) < len(df) * 0.5:
                        print("⚠️  Attention: Moins de 50% des valeurs sont numériques valides")
                        confirmer = input("Continuer quand même? (o/N): ").strip().lower()
                        if confirmer != 'o':
                            continue
                    
                    print(f"   Valeurs numériques: {len(valeurs_valides)}/{len(df)}")
                    print(f"   Min: {valeurs_valides.min():.2f}, Max: {valeurs_valides.max():.2f}")
                    return colonne_choisie
                    
                except Exception as e:
                    print(f"❌ Impossible de convertir en valeurs numériques: {e}")
                    continue
            else:
                print(f"❌ Numéro invalide. Choisissez entre 1 et {len(colonnes)}")
                
        except ValueError:
            print("❌ Veuillez entrer un numéro valide")


def configurer_prediction():
    """Configure les paramètres de prédiction"""
    print("\n⚙️  CONFIGURATION DE LA PRÉDICTION:")
    
    # Horizon de prédiction
    while True:
        try:
            horizon = input("👉 Nombre de périodes à prédire (défaut: 30): ").strip()
            
            if not horizon:
                horizon = 30
                break
            
            horizon = int(horizon)
            
            if 1 <= horizon <= 365:
                break
            else:
                print("❌ L'horizon doit être entre 1 et 365 jours")
                
        except ValueError:
            print("❌ Veuillez entrer un nombre entier")
    
    # Backend
    print("\nBackend de calcul:")
    print("   1. CPU (recommandé pour la plupart des cas)")
    print("   2. GPU (si CUDA disponible)")
    
    while True:
        choix_backend = input("👉 Choix (1-2, défaut: 1): ").strip()
        
        if not choix_backend or choix_backend == '1':
            backend = "cpu"
            break
        elif choix_backend == '2':
            backend = "gpu"
            print("⚠️  GPU sélectionné - Assurez-vous que CUDA est installé")
            break
        else:
            print("❌ Choisissez 1 ou 2")
    
    # Modèle TimesFM
    print("\nModèles TimesFM disponibles:")
    modeles = [
        ("google/timesfm-1.0-200m", "200M - Plus rapide, moins précis"),
        ("google/timesfm-1.0-200m-pytorch", "200M PyTorch - Équilibre performance/vitesse"),
    ]
    
    for i, (repo, description) in enumerate(modeles, 1):
        print(f"   {i}. {description}")
    
    while True:
        choix_modele = input(f"👉 Choix du modèle (1-{len(modeles)}, défaut: 2): ").strip()
        
        if not choix_modele or choix_modele == '2':
            model_repo = modeles[1][0]
            break
        
        try:
            index = int(choix_modele) - 1
            if 0 <= index < len(modeles):
                model_repo = modeles[index][0]
                break
            else:
                print(f"❌ Choisissez entre 1 et {len(modeles)}")
        except ValueError:
            print("❌ Veuillez entrer un numéro valide")
    
    # Mode de fonctionnement
    print("\nMode de fonctionnement:")
    print("   1. TimesFM réel (recommandé si installation complète)")
    print("   2. Mode simulation (pour tests rapides)")
    
    while True:
        choix_mode = input("👉 Choix (1-2, défaut: 1): ").strip()
        
        if not choix_mode or choix_mode == '1':
            simulation_mode = False
            break
        elif choix_mode == '2':
            simulation_mode = True
            print("⚠️  Mode simulation: Les prédictions seront simulées")
            break
        else:
            print("❌ Choisissez 1 ou 2")
    
    print("\n✅ CONFIGURATION:")
    print(f"   Horizon: {horizon} périodes")
    print(f"   Backend: {backend}")
    print(f"   Modèle: {model_repo}")
    print(f"   Mode: {'TimesFM réel' if not simulation_mode else 'Simulation'}")
    
    return {
        'horizon_len': horizon,
        'backend': backend,
        'model_repo': model_repo,
        'simulation_mode': simulation_mode
    }


def preparer_donnees_pour_prediction(df, colonne_valeurs):
    """Prépare les données pour TimesFM"""
    print("\n🔧 PRÉPARATION DES DONNÉES:")
    
    # Extraction et nettoyage des valeurs
    valeurs_raw = df[colonne_valeurs].copy()
    
    # Gestion des virgules décimales françaises
    if valeurs_raw.dtype == 'object':
        print("   Conversion des décimales (virgule → point)")
        valeurs_raw = valeurs_raw.astype(str).str.replace(',', '.')
    
    # Conversion numérique
    valeurs_numeriques = pd.to_numeric(valeurs_raw, errors='coerce')
    
    # Gestion des valeurs manquantes
    valeurs_manquantes = valeurs_numeriques.isna().sum()
    if valeurs_manquantes > 0:
        print(f"   Valeurs manquantes détectées: {valeurs_manquantes}")
        print("   Remplacement par interpolation linéaire")
        valeurs_numeriques = valeurs_numeriques.interpolate()
    
    # Conversion en array NumPy
    sales_array = valeurs_numeriques.to_numpy()
    
    print(f"✅ Données préparées:")
    print(f"   Points de données: {len(sales_array)}")
    print(f"   Min: {sales_array.min():.2f}")
    print(f"   Max: {sales_array.max():.2f}")
    print(f"   Moyenne: {sales_array.mean():.2f}")
    
    return sales_array


def executer_prediction(sales_array, config):
    """Exécute la prédiction avec TimesFM"""
    print("\n🚀 EXÉCUTION DE LA PRÉDICTION:")
    
    # Initialisation du prédicteur
    print("   Initialisation du modèle TimesFM...")
    predictor = TimesFMPredictor(
        horizon_len=config['horizon_len'],
        backend=config['backend'],
        model_repo=config['model_repo']
    )
    
    # Chargement du modèle
    print("   Chargement du modèle (peut prendre 1-2 minutes)...")
    predictor.load_model(simulation_mode=config['simulation_mode'])
    
    # Prédiction
    print("   Génération des prédictions...")
    result = predictor.predict_sales(sales_array)
    
    # Extraction des résultats
    predictions = result['predictions'][0]
    if hasattr(predictions, 'flatten'):
        predictions = predictions.flatten()
    else:
        predictions = np.array(predictions)
    
    return predictions, result


def afficher_resultats(predictions, result, config):
    """Affiche les résultats de prédiction"""
    print("\n" + "=" * 60)
    print("🎯 RÉSULTATS DES PRÉDICTIONS")
    print("=" * 60)
    
    # Informations du modèle
    print(f"\n🤖 MODÈLE UTILISÉ:")
    print(f"   Repository: {result.get('model_repo', 'N/A')}")
    print(f"   Mode: {'🎯 TimesFM Réel' if not config['simulation_mode'] else '🎲 Simulation'}")
    print(f"   Backend: {config['backend']}")
    print(f"   Horizon: {config['horizon_len']} périodes")
    
    # Prédictions détaillées
    print(f"\n📊 PRÉDICTIONS DÉTAILLÉES:")
    for i, pred in enumerate(predictions, 1):
        pred_value = float(pred.item() if hasattr(pred, 'item') else pred)
        print(f"   Période +{i:2d}: {pred_value:10.2f}")
    
    # Statistiques
    print(f"\n📈 STATISTIQUES:")
    print(f"   Total prédit: {predictions.sum():,.2f}")
    print(f"   Moyenne par période: {predictions.mean():.2f}")
    print(f"   Min prédit: {predictions.min():.2f}")
    print(f"   Max prédit: {predictions.max():.2f}")
    print(f"   Écart-type: {predictions.std():.2f}")
    
    # Informations simulation si applicable
    if 'simulation_stats' in result:
        stats = result['simulation_stats']
        print(f"\n🎲 DÉTAILS SIMULATION:")
        print(f"   Tendance détectée: {stats['detected_trend']:+.2f} par période")
        print(f"   Moyenne historique: {stats['mean_input']:.2f}")
        print(f"   Variabilité historique: {stats['std_input']:.2f}")


def sauvegarder_resultats(predictions, nom_fichier_source):
    """Propose de sauvegarder les résultats"""
    print(f"\n💾 SAUVEGARDE:")
    
    sauvegarder = input("Voulez-vous sauvegarder les résultats? (O/n): ").strip().lower()
    
    if sauvegarder in ['', 'o', 'oui']:
        # Nom de fichier de sortie basé sur le fichier source
        base_name = Path(nom_fichier_source).stem
        output_file = f"predictions_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        # Création du DataFrame de résultats
        df_results = pd.DataFrame({
            'periode': [f"+{i}" for i in range(1, len(predictions) + 1)],
            'prediction': [f"{pred:.2f}".replace('.', ',') for pred in predictions]
        })
        
        # Sauvegarde
        df_results.to_csv(output_file, sep=';', index=False)
        print(f"✅ Résultats sauvegardés dans: {output_file}")
        return output_file
    
    return None


def main():
    """Fonction principale du CLI"""
    try:
        # Titre
        afficher_titre()
        
        # 1. Sélection du fichier
        fichier_path = demander_fichier_donnees()
        
        # 2. Chargement des données
        df, sep, encoding = charger_donnees(fichier_path)
        if df is None:
            return False
        
        # 3. Sélection de la colonne
        colonne_valeurs = selectionner_colonne_valeurs(df)
        
        # 4. Configuration
        config = configurer_prediction()
        
        # 5. Préparation des données
        sales_array = preparer_donnees_pour_prediction(df, colonne_valeurs)
        
        # 6. Prédiction
        predictions, result = executer_prediction(sales_array, config)
        
        # 7. Affichage des résultats
        afficher_resultats(predictions, result, config)
        
        # 8. Sauvegarde optionnelle
        sauvegarder_resultats(predictions, fichier_path)
        
        print("\n" + "=" * 60)
        print("🎉 PRÉDICTION TERMINÉE AVEC SUCCÈS!")
        print("=" * 60)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n❌ Interruption utilisateur (Ctrl+C)")
        return False
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        print("💡 Vérifiez vos données et paramètres")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)