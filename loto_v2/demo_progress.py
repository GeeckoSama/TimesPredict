#!/usr/bin/env python3
"""
Demo des barres de progression Loto V2
Démonstrateur des nouvelles fonctionnalités de progress
"""

import sys
from pathlib import Path
import time
import random

# Ajouter le chemin des modules
sys.path.append(str(Path(__file__).parent))

from modules.progress import ProgressBar, loading_animation, simple_progress


def demo_progress_bar():
    """Démonstration de la barre de progression standard"""
    print("🔄 Demo ProgressBar Standard")
    
    # Simulation d'une tâche avec 20 étapes
    progress = ProgressBar(20, "📊 Calcul statistiques")
    
    for i in range(20):
        # Simulation de travail
        time.sleep(0.1)
        
        # Mise à jour description dynamique
        if i < 5:
            progress.set_description("📊 Chargement données")
        elif i < 15:
            progress.set_description("📊 Analyse fréquences")
        else:
            progress.set_description("📊 Sauvegarde")
        
        progress.update(1)
    
    print("✅ Demo ProgressBar terminée\n")


def demo_loading_animation():
    """Démonstration des animations de chargement"""
    print("🔄 Demo Loading Animation")
    
    loading_animation("Initialisation", 1.0)
    loading_animation("Préparation modèle", 1.5)
    loading_animation("Sauvegarde", 0.8)
    
    print("✅ Demo Animation terminée\n")


def demo_simple_progress():
    """Démonstration de la progress simple"""
    print("🔄 Demo Simple Progress")
    
    for i in range(11):
        progress_str = simple_progress(i, 10, "🎲 Prédictions: ")
        print(f"\r{progress_str}", end="", flush=True)
        time.sleep(0.2)
    
    print("\n✅ Demo Simple Progress terminée\n")


def demo_combined_workflow():
    """Démonstration d'un workflow complet avec progress"""
    print("🔄 Demo Workflow Complet")
    
    # 1. Chargement données avec animation
    loading_animation("Chargement données loto", 1.0)
    
    # 2. Calcul avec barre détaillée
    progress = ProgressBar(50, "🔄 Traitement")
    for i in range(50):
        if i < 10:
            progress.set_description("🔄 Lecture CSV")
        elif i < 30:
            progress.set_description("🔄 Calcul fréquences")
        elif i < 45:
            progress.set_description("🔄 Analyse statistique")
        else:
            progress.set_description("🔄 Finalisation")
        
        # Simulation travail variable
        time.sleep(random.uniform(0.02, 0.08))
        progress.update(1)
    
    # 3. Sauvegarde avec animation
    loading_animation("Sauvegarde résultats", 0.8)
    
    print("🎯 Résultat: 5616 tirages analysés")
    print("✅ Workflow terminé avec succès\n")


def main():
    """Lance toutes les démos"""
    print("=" * 50)
    print("🎯 DEMO BARRES DE PROGRESSION LOTO V2")
    print("=" * 50)
    print()
    
    # Lancer toutes les démos
    demo_progress_bar()
    demo_loading_animation()
    demo_simple_progress()
    demo_combined_workflow()
    
    print("🎉 Toutes les démos terminées!")


if __name__ == "__main__":
    main()