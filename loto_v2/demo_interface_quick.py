#!/usr/bin/env python3
"""
Démonstration rapide de l'interface améliorée Loto V2
Focus sur les améliorations visuelles sans long processing
"""

import sys
from pathlib import Path

# Ajouter le chemin des modules
sys.path.append(str(Path(__file__).parent))

from loto_v2 import LotoV2CLI
from modules.progress import UnifiedProgressBar
import time


def demo_quick_interface():
    """Démonstration rapide de l'interface améliorée"""
    print("=" * 60)
    print("🎯 LOTO V2 - INTERFACE AMÉLIORÉE (DÉMO RAPIDE)")
    print("=" * 60)
    print()
    
    cli = LotoV2CLI()
    
    # 1. Affichage du header et statut amélioré
    print("🎨 1. Menu avec statut détaillé")
    print("-" * 40)
    cli.display_header()
    cli.display_status()
    
    # 2. Démonstration de la détection GPU
    print("\n🚀 2. Détection automatique du hardware")
    print("-" * 40)
    fine_tuner = cli.fine_tuner
    config = fine_tuner.backend_config
    
    if config['use_gpu']:
        print(f"✅ GPU Mac M4 Pro détecté : {config['device'].upper()}")
        print(f"   Backend: {config['backend']}")
        print(f"   Batch size optimisé: {config['batch_size']}")
    else:
        print("🖥️  Mode CPU détecté")
    
    # 3. Démonstration des barres de progression unifiées
    print("\n📊 3. Barres de progression unifiées")
    print("-" * 40)
    print("Format : [████████░░░░] étape/max - Action en cours")
    print()
    
    # Exemple concret avec différentes phases
    progress = UnifiedProgressBar(6, "🔄 Traitement données")
    
    progress.set_step(1, "Chargement fichier CSV")
    time.sleep(0.3)
    
    progress.set_step(2, "Validation des données")
    time.sleep(0.2)
    
    progress.set_step(3, "Calcul des fréquences")
    progress.update_action("Analyse de 5616 tirages")
    time.sleep(0.4)
    
    progress.set_step(4, "Génération des probabilités")
    time.sleep(0.2)
    
    progress.set_step(5, "Préparation séquences TimesFM")
    time.sleep(0.3)
    
    progress.finish("Traitement terminé avec succès")
    
    # 4. Démonstration du menu complet
    print("\n📋 4. Menu principal amélioré")
    print("-" * 40)
    cli.display_menu()
    
    # 5. Statut final
    print("\n📊 5. Statut final mis à jour")
    print("-" * 40)
    cli.display_status()
    
    print("\n" + "=" * 60)
    print("🎉 DÉMONSTRATION INTERFACE TERMINÉE")
    print("=" * 60)
    print()
    print("💫 Améliorations démontrées :")
    print("   ✅ Détection automatique GPU Mac M4 Pro (MPS)")
    print("   ✅ Barres [████████░░░░] étape/max - action")
    print("   ✅ Statut détaillé avec dates et icônes")
    print("   ✅ Interface plus claire et informative")
    print("   ✅ Suivi en temps réel des opérations")
    print()
    print("⚡ Performances optimisées pour Apple Silicon")
    print("🎨 Design épuré et professionnel")


def demo_progress_variations():
    """Montre différentes variations de barres de progression"""
    print("\n🔧 BONUS : Variations de barres de progression")
    print("=" * 60)
    
    # Différents cas d'usage
    operations = [
        ("📥 Téléchargement", 3, ["Connexion au serveur", "Téléchargement 45MB", "Vérification intégrité"]),
        ("🤖 Modèle IA", 4, ["Init GPU", "Chargement poids", "Optimisation MPS", "Validation"]),
        ("🎯 Prédictions", 5, ["Analyse stats", "TimesFM forecast", "Pondération", "Validation", "Export"])
    ]
    
    for op_name, steps, actions in operations:
        progress = UnifiedProgressBar(steps, op_name)
        
        for i, action in enumerate(actions, 1):
            progress.set_step(i, action)
            time.sleep(0.2)
        
        progress.finish("Terminé")
        print()


def main():
    """Point d'entrée principal"""
    demo_quick_interface()
    demo_progress_variations()


if __name__ == "__main__":
    main()