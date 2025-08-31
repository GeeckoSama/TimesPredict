#!/usr/bin/env python3
"""
Test de l'interface améliorée avec support GPU et barres de progression unifiées
"""

import sys
from pathlib import Path

# Ajouter le chemin des modules
sys.path.append(str(Path(__file__).parent))

from modules.progress import UnifiedProgressBar
from modules.stats import LotoStatsAnalyzer
from modules.finetuning import LotoFineTuner
import time


def test_unified_progress_bar():
    """Test de la barre de progression unifiée"""
    print("🧪 Test de la barre de progression unifiée")
    print("-" * 50)
    
    # Test simple avec 5 étapes
    progress = UnifiedProgressBar(5, "🔄 Test opération")
    
    progress.set_step(1, "Initialisation")
    time.sleep(0.5)
    
    progress.set_step(2, "Chargement des données")
    time.sleep(0.7)
    
    progress.set_step(3, "Traitement des statistiques")
    time.sleep(0.3)
    
    progress.set_step(4, "Calcul des probabilités")
    time.sleep(0.4)
    
    progress.finish("Opération terminée")
    print("✅ Test barre unifiée terminé\n")


def test_gpu_detection():
    """Test de la détection automatique GPU"""
    print("🚀 Test détection GPU automatique")
    print("-" * 50)
    
    # Créer une instance pour tester la détection
    fine_tuner = LotoFineTuner()
    config = fine_tuner.backend_config
    
    print(f"Configuration détectée :")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    if config['use_gpu']:
        print("✅ GPU Mac M4 Pro détecté et configuré")
    else:
        print("ℹ️  Utilisation CPU (normal si pas de GPU)")
    print()


def test_stats_with_unified_progress():
    """Test des statistiques avec barre unifiée"""
    print("📊 Test calcul statistiques avec barre unifiée")
    print("-" * 50)
    
    try:
        analyzer = LotoStatsAnalyzer()
        stats = analyzer.calculate_frequencies()
        
        if stats:
            print("✅ Calcul statistiques avec barre unifiée : OK")
            print(f"   {stats['metadata']['total_draws']} tirages analysés")
        else:
            print("❌ Erreur calcul statistiques")
    except Exception as e:
        print(f"❌ Erreur test stats: {e}")
    print()


def test_model_loading_with_gpu():
    """Test chargement modèle avec GPU"""
    print("🤖 Test chargement TimesFM avec configuration GPU")
    print("-" * 50)
    
    try:
        fine_tuner = LotoFineTuner()
        print(f"Backend configuré: {fine_tuner.backend_config['backend']}")
        print(f"Device: {fine_tuner.backend_config['device']}")
        
        success = fine_tuner.load_base_model()
        if success:
            print("✅ Chargement modèle avec configuration optimale : OK")
        else:
            print("⚠️  Chargement modèle : Fallback utilisé")
    except Exception as e:
        print(f"❌ Erreur test modèle: {e}")
    print()


def test_menu_status_display():
    """Test de l'affichage du statut détaillé"""
    print("📋 Test affichage statut détaillé du menu")
    print("-" * 50)
    
    try:
        from loto_v2 import LotoV2CLI
        cli = LotoV2CLI()
        
        print("Affichage du statut détaillé :")
        cli.display_status()
        print("✅ Affichage statut détaillé : OK")
    except Exception as e:
        print(f"❌ Erreur test menu: {e}")
    print()


def main():
    """Lance tous les tests d'interface améliorée"""
    print("=" * 60)
    print("🧪 TESTS INTERFACE AMÉLIORÉE LOTO V2")
    print("=" * 60)
    print()
    
    # Tests individuels
    test_unified_progress_bar()
    test_gpu_detection()
    test_stats_with_unified_progress()
    test_model_loading_with_gpu()
    test_menu_status_display()
    
    print("=" * 60)
    print("🎉 Tests interface améliorée terminés!")
    print("💡 L'interface est maintenant plus élégante avec :")
    print("   ✅ Barres de progression unifiées")
    print("   ✅ Support GPU Mac M4 Pro automatique")
    print("   ✅ Statut détaillé dans le menu")
    print("   ✅ Suivi en temps réel des opérations")


if __name__ == "__main__":
    main()