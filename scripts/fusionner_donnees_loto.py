#!/usr/bin/env python3
"""
Fusion intelligente de tous les fichiers de données loto
Combine les différents formats et périodes en un fichier unique optimisé
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import re

def analyser_fichier_loto(filepath):
    """Analyse un fichier loto pour déterminer son format"""
    print(f"\n📁 Analyse: {filepath}")
    
    try:
        # Lecture des premières lignes pour analyser la structure
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # Détecter le séparateur et les colonnes
        if ';' in first_line:
            separator = ';'
        elif ',' in first_line:
            separator = ','
        else:
            separator = ';'  # Par défaut
        
        # Lire avec pandas pour analyser
        df = pd.read_csv(filepath, sep=separator, nrows=5)
        
        print(f"   Colonnes: {list(df.columns)}")
        print(f"   Séparateur: '{separator}'")
        print(f"   Lignes: {len(pd.read_csv(filepath, sep=separator))}")
        
        # Détecter le format selon les colonnes présentes
        colonnes = [col.lower().strip() for col in df.columns]
        
        if 'boule_6' in colonnes:
            format_type = "ancien_6_boules"  # Avant 2008, 6 boules + complémentaire
        elif 'boule_1_second_tirage' in colonnes:
            format_type = "moderne_double"   # Après 2019, avec 2ème tirage
        elif 'numero_chance' in colonnes and 'boule_5' in colonnes:
            format_type = "moderne_simple"   # 2008-2019, 5 boules + chance
        else:
            format_type = "inconnu"
            
        return {
            'filepath': filepath,
            'format': format_type,
            'separator': separator,
            'colonnes': df.columns.tolist(),
            'nb_lignes': len(pd.read_csv(filepath, sep=separator))
        }
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return None

def extraire_donnees_standardisees(info_fichier):
    """Extrait les données dans un format standardisé"""
    print(f"\n🔄 Traitement: {info_fichier['filepath']}")
    
    try:
        df = pd.read_csv(info_fichier['filepath'], sep=info_fichier['separator'])
        
        # Colonnes standardisées que nous voulons garder
        colonnes_std = ['date_tirage', 'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'numero_chance']
        
        data_std = []
        
        if info_fichier['format'] == "ancien_6_boules":
            print("   📅 Format ancien (6 boules) - Adaptation en cours...")
            
            for _, row in df.iterrows():
                try:
                    # Convertir la date
                    if 'date_de_tirage' in row:
                        date_str = str(row['date_de_tirage'])
                        if len(date_str) == 8:  # Format YYYYMMDD
                            date_obj = datetime.strptime(date_str, '%Y%m%d')
                        else:
                            continue  # Skip si format de date non reconnu
                    else:
                        continue
                    
                    # Pour l'ancien format, on prend les 5 premières boules + on simule un numéro chance
                    # (il n'y avait pas de numéro chance avant 2008)
                    boules = [row.get(f'boule_{i}', None) for i in range(1, 6)]
                    
                    # Vérifier que toutes les boules sont valides
                    if all(pd.notna(b) and 1 <= float(b) <= 49 for b in boules):
                        # Simuler un numéro chance basé sur la 6ème boule (modulo 10 + 1)
                        boule_6 = row.get('boule_6', 25)  # Default si manquant
                        numero_chance = int(float(boule_6)) % 10 + 1
                        
                        data_std.append({
                            'date_tirage': date_obj.strftime('%d/%m/%Y'),
                            'boule_1': int(float(boules[0])),
                            'boule_2': int(float(boules[1])),
                            'boule_3': int(float(boules[2])),
                            'boule_4': int(float(boules[3])),
                            'boule_5': int(float(boules[4])),
                            'numero_chance': numero_chance,
                            'source': 'ancien_format'
                        })
                except Exception as e:
                    continue  # Skip cette ligne si erreur
                    
        elif info_fichier['format'] in ["moderne_simple", "moderne_double"]:
            print("   🆕 Format moderne - Extraction directe...")
            
            for _, row in df.iterrows():
                try:
                    # Convertir la date
                    date_col = None
                    for col in ['date_de_tirage', 'date_tirage']:
                        if col in row:
                            date_col = col
                            break
                    
                    if not date_col:
                        continue
                        
                    date_str = str(row[date_col])
                    
                    # Gérer différents formats de date
                    if '/' in date_str:  # DD/MM/YYYY
                        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                    elif '-' in date_str:  # YYYY-MM-DD
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    else:
                        continue
                    
                    # Extraire les boules et numéro chance
                    boules = [row.get(f'boule_{i}', None) for i in range(1, 6)]
                    numero_chance = row.get('numero_chance', None)
                    
                    # Vérifier la validité
                    if all(pd.notna(b) and 1 <= float(b) <= 49 for b in boules) and \
                       pd.notna(numero_chance) and 1 <= float(numero_chance) <= 10:
                        
                        data_std.append({
                            'date_tirage': date_obj.strftime('%d/%m/%Y'),
                            'boule_1': int(float(boules[0])),
                            'boule_2': int(float(boules[1])),
                            'boule_3': int(float(boules[2])),
                            'boule_4': int(float(boules[3])),
                            'boule_5': int(float(boules[4])),
                            'numero_chance': int(float(numero_chance)),
                            'source': 'moderne_format'
                        })
                except Exception as e:
                    continue  # Skip cette ligne si erreur
        
        print(f"   ✅ Extraites: {len(data_std)} entrées valides")
        return data_std
        
    except Exception as e:
        print(f"   ❌ Erreur extraction: {e}")
        return []

def fusionner_donnees_loto():
    """Fonction principale de fusion des données loto"""
    print("🎰 FUSION DES DONNÉES LOTO HISTORIQUES")
    print("=" * 50)
    
    # 1. Analyser tous les fichiers
    fichiers_loto = [
        "data/raw/loto.csv",
        "data/raw/loto2017.csv", 
        "data/raw/loto2019.csv",
        "data/raw/loto_201902.csv",
        "data/raw/loto_201911.csv"
    ]
    
    analyses = []
    for fichier in fichiers_loto:
        if os.path.exists(fichier):
            analyse = analyser_fichier_loto(fichier)
            if analyse:
                analyses.append(analyse)
    
    print(f"\n📊 RÉSUMÉ DES FICHIERS ANALYSÉS:")
    total_lignes = 0
    for analyse in analyses:
        print(f"   📁 {Path(analyse['filepath']).name}: {analyse['nb_lignes']} lignes ({analyse['format']})")
        total_lignes += analyse['nb_lignes']
    
    print(f"   📈 TOTAL: {total_lignes} tirages potentiels")
    
    # 2. Extraire et fusionner toutes les données
    print(f"\n🔄 EXTRACTION ET STANDARDISATION:")
    
    toutes_donnees = []
    for analyse in analyses:
        donnees = extraire_donnees_standardisees(analyse)
        toutes_donnees.extend(donnees)
    
    # 3. Convertir en DataFrame et nettoyer
    print(f"\n🧹 NETTOYAGE ET DÉDOUBLONNAGE:")
    df_complet = pd.DataFrame(toutes_donnees)
    
    if len(df_complet) == 0:
        print("❌ Aucune donnée extraite !")
        return
    
    # Convertir les dates pour le tri
    df_complet['date_obj'] = pd.to_datetime(df_complet['date_tirage'], format='%d/%m/%Y')
    
    # Supprimer les doublons basés sur la date
    avant_dedup = len(df_complet)
    df_complet = df_complet.drop_duplicates(subset=['date_tirage'], keep='first')
    apres_dedup = len(df_complet)
    
    print(f"   🗑️  Doublons supprimés: {avant_dedup - apres_dedup}")
    
    # Trier par date
    df_complet = df_complet.sort_values('date_obj')
    
    print(f"   📅 Période couverte: {df_complet['date_obj'].min().strftime('%d/%m/%Y')} → {df_complet['date_obj'].max().strftime('%d/%m/%Y')}")
    print(f"   📊 Données finales: {len(df_complet)} tirages uniques")
    
    # 4. Sauvegarder le fichier fusionné
    output_file = "data/raw/loto_complet_fusionne.csv"
    
    # Préparer les données finales (sans la colonne date_obj temporaire)
    df_final = df_complet[['date_tirage', 'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'numero_chance', 'source']].copy()
    
    # Sauvegarder avec le format attendu par le système
    df_final.columns = ['date_de_tirage', 'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'numero_chance', 'source_donnees']
    df_final.to_csv(output_file, sep=';', index=False, encoding='utf-8')
    
    print(f"\n💾 SAUVEGARDE:")
    print(f"   📁 Fichier créé: {output_file}")
    print(f"   📊 {len(df_final)} tirages sauvegardés")
    
    # 5. Statistiques finales
    print(f"\n📈 STATISTIQUES FINALES:")
    print(f"   🎯 Période totale: {(df_complet['date_obj'].max() - df_complet['date_obj'].min()).days} jours")
    print(f"   📅 Années couvertes: {df_complet['date_obj'].dt.year.nunique()} années différentes")
    print(f"   🔥 Boule la plus fréquente: {df_final[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].values.flatten().mode()[0] if len(df_final) > 0 else 'N/A'}")
    print(f"   🎲 Chance la plus fréquente: {df_final['numero_chance'].mode()[0] if len(df_final) > 0 else 'N/A'}")
    
    # Afficher quelques lignes d'exemple
    print(f"\n📋 APERÇU DES DONNÉES (5 premiers tirages):")
    print(df_final.head().to_string(index=False))
    
    return output_file

if __name__ == "__main__":
    fichier_fusionne = fusionner_donnees_loto()
    print(f"\n🎉 FUSION TERMINÉE ! Utilisez le fichier: {fichier_fusionne}")