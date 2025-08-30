#!/usr/bin/env python3
"""
CLI Principal pour le Prédicteur Loto TimesFM
Interface interactive pour analyser et prédire les tirages de loto français
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Ajouter les chemins nécessaires
sys.path.append("src")

# Imports des modules spécialisés loto
try:
    from loto_predict.data.loto_data_processor import LotoDataProcessor
    from loto_predict.models.multi_timesfm_predictor import MultiTimesFMPredictor
    from loto_predict.analysis.loto_stat_analyzer import LotoStatAnalyzer
    from loto_predict.optimization.combination_generator import CombinationGenerator
    from loto_predict.validation.backtest_validator import BacktestValidator
    from scripts.integration_cli_echantillonnage import executer_echantillonnage_massif, configurer_echantillonnage_massif
except ImportError as e:
    print(f"❌ Erreur d'import des modules loto: {e}")
    print("💡 Assurez-vous que tous les modules sont correctement installés")
    sys.exit(1)


def afficher_titre_loto():
    """Affiche le titre de l'application loto"""
    print("\n" + "=" * 70)
    print("🎰 LOTO TIMESFM CLI - PRÉDICTEUR INTELLIGENT")
    print("=" * 70)
    print("Analyse et prédiction des tirages de loto français avec TimesFM")
    print("🚨 IMPORTANT: Usage éducatif - Aucune garantie de gain")
    print("=" * 70)


def detecter_fichier_loto():
    """Détecte automatiquement le fichier de données loto"""
    fichiers_possibles = [
        "data/raw/loto_complet_fusionne.csv",  # NOUVEAU : Dataset fusionné complet
        "data/raw/loto_201911.csv",
        "data/raw/loto.csv", 
        "loto_201911.csv",
        "loto.csv"
    ]
    
    print("🔍 Recherche du fichier de données loto...")
    
    for fichier in fichiers_possibles:
        if Path(fichier).exists():
            print(f"✅ Fichier trouvé: {fichier}")
            return str(Path(fichier))
    
    # Demander à l'utilisateur
    while True:
        fichier = input("\n👉 Chemin vers votre fichier de données loto (.csv): ").strip()
        
        if not fichier:
            print("❌ Veuillez entrer un chemin de fichier")
            continue
            
        fichier_path = Path(fichier)
        
        if not fichier_path.exists():
            print(f"❌ Fichier non trouvé: {fichier}")
            continue
            
        if not fichier.lower().endswith('.csv'):
            print("❌ Le fichier doit être au format CSV")
            continue
            
        return str(fichier_path)


def configurer_prediction_loto():
    """Configure les paramètres de prédiction loto"""
    print("\n⚙️  CONFIGURATION DE LA PRÉDICTION LOTO:")
    
    # Nombre de combinaisons
    while True:
        try:
            nb_combos = input("👉 Nombre de combinaisons à générer (défaut: 5): ").strip()
            
            if not nb_combos:
                nb_combos = 5
                break
            
            nb_combos = int(nb_combos)
            
            if 1 <= nb_combos <= 20:
                break
            else:
                print("❌ Le nombre doit être entre 1 et 20")
                
        except ValueError:
            print("❌ Veuillez entrer un nombre entier")
    
    # Backend TimesFM
    print("\nBackend de calcul:")
    print("   1. CPU (recommandé et stable)")
    print("   2. GPU (plus rapide si CUDA disponible)")
    
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
        ("google/timesfm-1.0-200m-pytorch", "200M v1.0 PyTorch - Rapide et stable"),
        ("google/timesfm-2.0-500m-pytorch", "500M v2.0 PyTorch - Plus précis (RECOMMANDÉ)"),
    ]
    
    for i, (repo, description) in enumerate(modeles, 1):
        print(f"   {i}. {description}")
    
    while True:
        choix_modele = input(f"👉 Choix du modèle (1-{len(modeles)}, défaut: 2): ").strip()
        
        if not choix_modele or choix_modele == '2':
            model_repo = modeles[1][0]  # TimesFM 2.0 par défaut
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
    
    # Configuration du contexte historique
    print(f"\nTaille du contexte historique:")
    print(f"   Le dataset contient 5616 tirages depuis 1976")
    
    # Calculer les limites selon le modèle sélectionné
    if "2.0" in model_repo and "500m" in model_repo:
        max_context = 2048
        print(f"   Modèle TimesFM 2.0 : capacité maximale de 2048 tirages")
    else:
        max_context = 512
        print(f"   Modèle TimesFM 1.0 : capacité maximale de 512 tirages")
    
    context_options = [
        (10, f"10% du dataset (~{int(5616 * 0.1)} tirages, {int(5616 * 0.1 * 7/365):.0f} ans)"),
        (25, f"25% du dataset (~{int(5616 * 0.25)} tirages, {int(5616 * 0.25 * 7/365):.0f} ans)"),
        (50, f"50% du dataset (~{int(5616 * 0.5)} tirages, {int(5616 * 0.5 * 7/365):.0f} ans)"),
        (100, f"100% du dataset (limité à {min(5616, max_context)} par le modèle)")
    ]
    
    for i, (percentage, description) in enumerate(context_options, 1):
        print(f"   {i}. {description}")
    
    while True:
        choix_context = input("👉 Choix du contexte (1-4, défaut: 4): ").strip()
        
        if not choix_context or choix_context == '4':
            context_percentage = 100
            break
        
        try:
            index = int(choix_context) - 1
            if 0 <= index < len(context_options):
                context_percentage = context_options[index][0]
                break
            else:
                print(f"❌ Choisissez entre 1 et {len(context_options)}")
        except ValueError:
            print("❌ Veuillez entrer un numéro valide")

    # Mode de fonctionnement
    print("\nMode de fonctionnement:")
    print("   1. TimesFM réel (prédictions avec IA)")
    print("   2. Mode simulation (tests rapides sans IA)")
    
    while True:
        choix_mode = input("👉 Choix (1-2, défaut: 1): ").strip()
        
        if not choix_mode or choix_mode == '1':
            simulation_mode = False
            break
        elif choix_mode == '2':
            simulation_mode = True
            print("⚠️  Mode simulation: Prédictions simulées pour tests")
            break
        else:
            print("❌ Choisissez 1 ou 2")
    
    # Calculer le contexte effectif
    if context_percentage == 100:
        context_length = min(5616, max_context)
    else:
        context_length = min(int(5616 * context_percentage / 100), max_context)
    
    print("\n✅ CONFIGURATION:")
    print(f"   Combinaisons: {nb_combos}")
    print(f"   Backend: {backend}")
    print(f"   Modèle: {model_repo}")
    print(f"   Contexte: {context_percentage}% (~{context_length} tirages)")
    print(f"   Mode: {'TimesFM Réel' if not simulation_mode else 'Simulation'}")
    
    return {
        'nb_combinations': nb_combos,
        'backend': backend,
        'context_percentage': context_percentage,
        'context_length': context_length,
        'model_repo': model_repo,
        'simulation_mode': simulation_mode
    }


def executer_analyse_complete(data_processor: LotoDataProcessor):
    """Exécute l'analyse statistique complète"""
    print("\n🔍 ANALYSE STATISTIQUE COMPLÈTE")
    print("=" * 50)
    
    # Traitement des données
    processed_result = data_processor.process_data()
    processed_data = processed_result['data']
    
    # Analyse statistique
    analyzer = LotoStatAnalyzer(processed_data)
    analysis_results = analyzer.run_full_analysis()
    
    # Affichage des insights clés
    print("\n📈 INSIGHTS CLÉS:")
    
    if 'frequencies' in analysis_results:
        freq = analysis_results['frequencies']
        hot_numbers = freq.get('hot_numbers', [])
        cold_numbers = freq.get('cold_numbers', [])
        
        print(f"   🔥 Numéros chauds: {hot_numbers[:10]}")
        print(f"   🧊 Numéros froids: {cold_numbers[:10]}")
    
    if 'temporal' in analysis_results:
        temporal = analysis_results['temporal']
        if 'day_of_week_patterns' in temporal:
            dow_patterns = temporal['day_of_week_patterns']
            if dow_patterns:
                best_day = max(dow_patterns.keys(), key=lambda d: dow_patterns[d]['count'])
                print(f"   📅 Jour le plus fréquent: {['Lun','Mar','Mer','Jeu','Ven','Sam','Dim'][best_day]}")
    
    return analysis_results


def executer_prediction_loto(data_processor: LotoDataProcessor, config: dict):
    """Exécute la prédiction avec TimesFM"""
    print("\n🚀 PRÉDICTION AVEC TIMESFM MULTI-MODÈLES")
    print("=" * 50)
    
    # Traiter les données et créer les séries temporelles
    print("📊 Préparation des séries temporelles...")
    processed_result = data_processor.process_data()
    time_series = data_processor.create_time_series()
    
    # Initialiser le prédicteur multi-TimesFM
    print(f"🤖 Initialisation de {6} modèles TimesFM...")
    predictor = MultiTimesFMPredictor(
        model_repo=config['model_repo'],
        backend=config['backend'],
        horizon_len=1
    )
    
    # Charger les modèles
    print("⏳ Chargement des modèles (peut prendre 2-3 minutes)...")
    success = predictor.load_models(simulation_mode=config['simulation_mode'])
    
    if not success:
        print("❌ Échec du chargement des modèles")
        return None
    
    # Faire la prédiction avec le contexte configuré
    print(f"🎯 Génération de la prédiction (contexte: {config['context_length']} tirages)...")
    prediction_result = predictor.predict_next_combination(
        time_series, 
        context_length=config['context_length']
    )
    
    return prediction_result


def generer_combinaisons_optimisees(prediction_result: dict, 
                                   analysis_results: dict, 
                                   config: dict,
                                   processed_data):
    """Génère des combinaisons optimisées"""
    print("\n🎯 GÉNÉRATION DE COMBINAISONS OPTIMISÉES")
    print("=" * 50)
    
    # Initialiser le générateur
    generator = CombinationGenerator(processed_data)
    
    # Stratégies à utiliser
    strategies = ['timesfm_direct', 'statistical_weighted', 'hybrid_optimized']
    
    # Générer les combinaisons
    combinations_result = generator.generate_combinations(
        timesfm_predictions=prediction_result,
        statistical_analysis=analysis_results,
        num_combinations=config['nb_combinations'],
        strategies=strategies
    )
    
    return combinations_result


def afficher_predictions(prediction_result: dict, combinations_result: dict):
    """Affiche les résultats de prédiction"""
    print("\n" + "=" * 70)
    print("🎰 RÉSULTATS DES PRÉDICTIONS LOTO")
    print("=" * 70)
    
    # Prédiction TimesFM directe
    if 'combination' in prediction_result:
        direct_combo = prediction_result['combination']
        print(f"\n🤖 PRÉDICTION TIMESFM DIRECTE:")
        print(f"   Boules: {direct_combo['boules']}")
        print(f"   Numéro chance: {direct_combo['numero_chance']}")
        
        if 'metadata' in prediction_result:
            metadata = prediction_result['metadata']
            print(f"   Confiance moyenne: {metadata.get('confiance_moyenne', 0):.1%}")
            print(f"   Somme des boules: {metadata.get('somme_boules', 0):.0f}")
    
    # Combinaisons optimisées
    if 'combinations' in combinations_result:
        print(f"\n🎯 COMBINAISONS OPTIMISÉES ({len(combinations_result['combinations'])}):")
        
        for i, combo in enumerate(combinations_result['combinations'], 1):
            print(f"\n   #{i} - Score: {combo['score']:.3f} - Méthode: {combo['method']}")
            print(f"       Boules: {combo['boules']}")
            print(f"       Chance: {combo['numero_chance']}")
            print(f"       Confiance: {combo['confidence']:.1%}")
    
    # Métadonnées de génération
    if 'generation_stats' in combinations_result:
        stats = combinations_result['generation_stats']
        print(f"\n📊 STATISTIQUES DE GÉNÉRATION:")
        print(f"   Total générées: {stats['total_generated']}")
        print(f"   Uniques retenues: {stats['unique_combinations']}")
        print(f"   Stratégies utilisées: {', '.join(stats['strategies_used'])}")


def sauvegarder_predictions(prediction_result: dict, combinations_result: dict, config: dict):
    """Sauvegarde les prédictions"""
    print(f"\n💾 SAUVEGARDE DES PRÉDICTIONS:")
    
    sauver = input("Voulez-vous sauvegarder les prédictions? (O/n): ").strip().lower()
    
    if sauver in ['', 'o', 'oui']:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"loto_predictions_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"PRÉDICTIONS LOTO - {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
                f.write("=" * 60 + "\n\n")
                
                # Configuration utilisée
                f.write(f"CONFIGURATION:\n")
                f.write(f"  Modèle: {config['model_repo']}\n")
                f.write(f"  Backend: {config['backend']}\n")
                f.write(f"  Mode: {'TimesFM' if not config['simulation_mode'] else 'Simulation'}\n\n")
                
                # Prédiction directe
                if 'combination' in prediction_result:
                    combo = prediction_result['combination']
                    f.write(f"PRÉDICTION TIMESFM DIRECTE:\n")
                    f.write(f"  Boules: {combo['boules']}\n")
                    f.write(f"  Numéro chance: {combo['numero_chance']}\n\n")
                
                # Combinaisons optimisées
                if 'combinations' in combinations_result:
                    f.write(f"COMBINAISONS OPTIMISÉES:\n")
                    for i, combo in enumerate(combinations_result['combinations'], 1):
                        f.write(f"  #{i} - Score: {combo['score']:.3f}\n")
                        f.write(f"      Boules: {combo['boules']}\n")
                        f.write(f"      Chance: {combo['numero_chance']}\n")
                        f.write(f"      Méthode: {combo['method']}\n\n")
                
                f.write("\nIMPORTANT: Prédictions à usage éducatif uniquement.\n")
                f.write("Aucune garantie de gain. Jouez avec modération.\n")
            
            print(f"✅ Prédictions sauvegardées: {filename}")
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")


def executer_backtest(data_processor: LotoDataProcessor, config: dict):
    """Exécute un backtest pour valider les performances"""
    print("\n🧪 BACKTEST - VALIDATION DES PERFORMANCES")
    print("=" * 50)
    
    confirmer = input("Le backtest peut prendre 10-20 minutes. Continuer? (o/N): ").strip().lower()
    
    if confirmer != 'o':
        print("Backtest annulé")
        return
    
    # Initialiser le validateur
    validator = BacktestValidator(data_processor.processed_data)
    
    # Initialiser le prédicteur
    predictor = MultiTimesFMPredictor(
        model_repo=config['model_repo'],
        backend=config['backend'],
        horizon_len=1
    )
    
    # Exécuter le backtest
    print("🔄 Exécution du backtest (cela peut prendre du temps)...")
    results = validator.run_backtest(
        predictor=predictor,
        test_period_days=50,  # Tester sur 50 tirages
        prediction_frequency=5  # Une prédiction tous les 5 tirages
    )
    
    # Sauvegarder les résultats
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    backtest_file = f"loto_backtest_{timestamp}.json"
    validator.save_results(backtest_file)
    
    # Afficher les recommandations
    recommendations = validator.get_prediction_recommendations()
    
    print(f"\n💡 RECOMMANDATIONS:")
    print(f"   Performance du modèle: {recommendations.get('model_performance', 'unknown')}")
    print(f"   Niveau de confiance: {recommendations.get('confidence_level', 'low')}")
    print(f"   Usage recommandé: {recommendations.get('recommended_usage', 'cautious')}")
    
    for suggestion in recommendations.get('suggestions', []):
        print(f"   • {suggestion}")


def menu_principal():
    """Menu principal de l'application"""
    print("\n📋 MENU PRINCIPAL:")
    print("   1. 🎯 Générer des prédictions loto")
    print("   2. 🎲 Échantillonnage massif (1000 prédictions → meilleure combo)")
    print("   3. 🔍 Analyse statistique complète")
    print("   4. 🧪 Backtest / Validation des performances")
    print("   5. 📊 Analyse + Prédictions (complet)")
    print("   6. ❌ Quitter")
    
    while True:
        choix = input("\n👉 Votre choix (1-6): ").strip()
        
        if choix in ['1', '2', '3', '4', '5', '6']:
            return int(choix)
        else:
            print("❌ Choix invalide. Sélectionnez entre 1 et 6.")


def main():
    """Fonction principale du CLI loto"""
    try:
        # Titre
        afficher_titre_loto()
        
        # Détection du fichier de données
        fichier_loto = detecter_fichier_loto()
        
        # Chargement des données
        print("\n📂 CHARGEMENT DES DONNÉES LOTO:")
        data_processor = LotoDataProcessor(fichier_loto)
        raw_data = data_processor.load_data()
        
        if raw_data is None or len(raw_data) == 0:
            print("❌ Impossible de charger les données loto")
            return False
        
        # Menu principal
        while True:
            choix = menu_principal()
            
            if choix == 1:  # Prédictions uniquement
                config = configurer_prediction_loto()
                
                prediction_result = executer_prediction_loto(data_processor, config)
                if prediction_result:
                    # Analyse rapide pour le générateur
                    processed_result = data_processor.process_data()
                    analyzer = LotoStatAnalyzer(processed_result['data'])
                    quick_analysis = analyzer.run_full_analysis()
                    
                    combinations_result = generer_combinaisons_optimisees(
                        prediction_result, quick_analysis, config, processed_result['data']
                    )
                    
                    afficher_predictions(prediction_result, combinations_result)
                    sauvegarder_predictions(prediction_result, combinations_result, config)
            
            elif choix == 2:  # Échantillonnage massif
                nb_echantillons = configurer_echantillonnage_massif()
                
                # Configuration de base (hérite de la config normale)
                config = configurer_prediction_loto()
                config['nb_echantillons'] = nb_echantillons
                
                # Exécuter échantillonnage
                resultat_massif = executer_echantillonnage_massif(data_processor, config)
                
                if resultat_massif and resultat_massif['success']:
                    print("\n🎉 ÉCHANTILLONNAGE MASSIF TERMINÉ!")
                    combo = resultat_massif['combination']
                    stats = resultat_massif['statistics']
                    
                    print(f"\n🏆 RÉSULTAT FINAL:")
                    print(f"   🎲 Combinaison recommandée: {combo['boules']} + {combo['numero_chance']}")
                    print(f"   📊 Basée sur {stats['total_predictions']} prédictions TimesFM")
                    print(f"   ✅ Convergence: {stats['convergence']:.1f}%")
                    print(f"   🔄 Diversité: {stats['diversity']:.1f}%")
                    
                    # Sauvegarder si souhaité
                    sauvegarder = input("\nSauvegarder ce résultat? (O/n): ").strip().lower()
                    if sauvegarder in ['o', 'oui', '']:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                        filename = f"loto_echantillonnage_massif_{timestamp}.txt"
                        
                        try:
                            with open(filename, 'w', encoding='utf-8') as f:
                                f.write(f"ÉCHANTILLONNAGE MASSIF LOTO - {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
                                f.write("=" * 70 + "\n\n")
                                
                                f.write(f"CONFIGURATION:\n")
                                f.write(f"  Nombre de prédictions: {stats['total_predictions']}\n")
                                f.write(f"  Modèle: {config['model_repo']}\n")
                                f.write(f"  Backend: {config['backend']}\n")
                                f.write(f"  Contexte: {config['context_length']} tirages\n\n")
                                
                                f.write(f"COMBINAISON RECOMMANDÉE:\n")
                                f.write(f"  Boules: {combo['boules']}\n")
                                f.write(f"  Numéro chance: {combo['numero_chance']}\n")
                                f.write(f"  Fréquence: {combo['frequence']}x sur {stats['total_predictions']} ({combo['pourcentage']:.1f}%)\n\n")
                                
                                if 'top_combinations' in resultat_massif:
                                    f.write(f"TOP 5 COMBINAISONS LES PLUS FRÉQUENTES:\n")
                                    for i, (combo_tuple, freq) in enumerate(resultat_massif['top_combinations'], 1):
                                        boules = sorted(list(combo_tuple[:-1]))
                                        chance = combo_tuple[-1]
                                        pct = freq / stats['total_predictions'] * 100
                                        f.write(f"  #{i}: {boules} + {chance} → {freq}x ({pct:.1f}%)\n")
                                
                                f.write(f"\nSTATISTIQUES:\n")
                                f.write(f"  Convergence: {stats['convergence']:.1f}%\n")
                                f.write(f"  Diversité: {stats['diversity']:.1f}%\n")
                                f.write(f"  Combinaisons uniques: {stats['unique_combinations']}/{stats['total_predictions']}\n")
                                
                                f.write("\nIMPORTANT: Prédictions à usage éducatif uniquement.\n")
                                f.write("Aucune garantie de gain. Jouez avec modération.\n")
                            
                            print(f"✅ Résultat sauvegardé: {filename}")
                            
                        except Exception as e:
                            print(f"❌ Erreur lors de la sauvegarde: {e}")
                else:
                    print("❌ Échec de l'échantillonnage massif")
            
            elif choix == 3:  # Analyse statistique
                analysis_results = executer_analyse_complete(data_processor)
                
                # Optionnel: sauvegarder l'analyse
                sauver_analyse = input("\nSauvegarder l'analyse? (o/N): ").strip().lower()
                if sauver_analyse == 'o':
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                    import json
                    with open(f"loto_analysis_{timestamp}.json", 'w') as f:
                        # Convertir les objets non-sérialisables
                        json_safe = {}
                        for key, value in analysis_results.items():
                            try:
                                json.dumps(value)  # Test de sérialisabilité
                                json_safe[key] = value
                            except (TypeError, ValueError):
                                json_safe[key] = str(value)  # Convertir en string si problème
                        json.dump(json_safe, f, indent=2, ensure_ascii=False)
                    print(f"✅ Analyse sauvegardée: loto_analysis_{timestamp}.json")
            
            elif choix == 4:  # Backtest
                config = configurer_prediction_loto()
                executer_backtest(data_processor, config)
            
            elif choix == 5:  # Analyse + Prédictions complètes
                print("\n🚀 ANALYSE ET PRÉDICTIONS COMPLÈTES")
                config = configurer_prediction_loto()
                
                # Analyse complète
                analysis_results = executer_analyse_complete(data_processor)
                
                # Prédictions
                prediction_result = executer_prediction_loto(data_processor, config)
                
                if prediction_result:
                    combinations_result = generer_combinaisons_optimisees(
                        prediction_result, analysis_results, config, data_processor.processed_data
                    )
                    
                    afficher_predictions(prediction_result, combinations_result)
                    sauvegarder_predictions(prediction_result, combinations_result, config)
            
            elif choix == 6:  # Quitter
                print("\n👋 Au revoir !")
                print("🎰 Rappel: Jouez avec modération, les prédictions sont éducatives.")
                break
            
            # Demander si continuer
            if choix != 6:
                continuer = input("\n🔄 Retourner au menu principal? (O/n): ").strip().lower()
                if continuer == 'n':
                    print("\n👋 Au revoir !")
                    break
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n❌ Interruption utilisateur (Ctrl+C)")
        return False
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        print("💡 Vérifiez vos données et la configuration")
        return False


if __name__ == "__main__":
    print("🎰 Démarrage du Prédicteur Loto TimesFM...")
    
    # Avertissement important
    print("\n" + "⚠️ " * 20)
    print("AVERTISSEMENT IMPORTANT:")
    print("Ce logiciel est à usage ÉDUCATIF et de RECHERCHE uniquement.")
    print("Les prédictions ne garantissent AUCUN gain au loto.")
    print("Le loto est un jeu de hasard. Jouez avec modération.")
    print("⚠️ " * 20)
    
    accepter = input("\nAcceptez-vous ces conditions? (o/N): ").strip().lower()
    
    if accepter == 'o':
        success = main()
        sys.exit(0 if success else 1)
    else:
        print("\n👋 Programme arrêté.")
        sys.exit(0)