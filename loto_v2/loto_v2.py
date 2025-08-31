#!/usr/bin/env python3
"""
Loto V2 CLI - Interface Simplifiée
Prédictions pondérées par fréquences historiques + fine-tuning TimesFM
"""

import sys
from pathlib import Path

# Ajouter le chemin des modules
sys.path.append(str(Path(__file__).parent))

from modules.storage import LotoStorage
from modules.stats import LotoStatsAnalyzer
from modules.finetuning import LotoFineTuner
from modules.prediction import LotoPredictor
from modules.validation import LotoValidator
from modules.progress import loading_animation


class LotoV2CLI:
    """Interface CLI simplifiée pour Loto V2"""
    
    def __init__(self):
        self.storage = LotoStorage()
        self.stats_analyzer = LotoStatsAnalyzer()
        self.fine_tuner = LotoFineTuner()
        self.predictor = LotoPredictor()
        self.validator = LotoValidator()
        
    def display_header(self):
        """Affiche l'en-tête du programme"""
        print("\n" + "=" * 50)
        print("🎯 LOTO V2 - Prédicteur Simplifié")
        print("=" * 50)
    
    def display_status(self):
        """Affiche le statut détaillé des opérations"""
        status = self.storage.get_status()
        
        print("\n📊 Statut des opérations:")
        
        # Statut des statistiques
        stats_status = status.get('stats', None)
        if stats_status and stats_status != '❌':
            # Extraire la date du format "✅ YYYY-MM-DDTHH:MM:SS"
            if isinstance(stats_status, str) and stats_status.startswith('✅'):
                date_part = stats_status[2:].split('T')[0] if 'T' in stats_status else stats_status[2:]
                print(f"   📈 Statistiques: ✅ Calculées le {date_part}")
            else:
                print(f"   📈 Statistiques: ✅ Disponibles")
        else:
            print(f"   📈 Statistiques: ❌ Non calculées")
        
        # Statut du fine-tuning
        ft_status = status.get('finetuning', None)
        if ft_status and ft_status != '❌':
            if isinstance(ft_status, str) and ft_status.startswith('✅'):
                date_part = ft_status[2:].split('T')[0] if 'T' in ft_status else ft_status[2:]
                print(f"   🤖 Fine-tuning: ✅ Effectué le {date_part}")
            else:
                print(f"   🤖 Fine-tuning: ✅ Modèle prêt")
        else:
            print(f"   🤖 Fine-tuning: ❌ Non effectué")
        
        # Statut hardware détecté
        hw_config = self.fine_tuner.backend_config
        print(f"\n🔧 Hardware détecté:")
        if hw_config['gpu_type'] == 'nvidia_cuda':
            print(f"   🚀 GPU: {hw_config['device_name']}")
            if 'memory_gb' in hw_config:
                print(f"   💾 VRAM: {hw_config['memory_gb']:.1f} GB")
            print(f"   ⚡ CUDA avec batch size {hw_config['batch_size']}")
        elif hw_config['gpu_type'] == 'apple_mps':
            print(f"   🍎 GPU: {hw_config['device_name']}")
            print(f"   ⚡ MPS avec batch size {hw_config['batch_size']}")
        else:
            print(f"   🖥️  CPU: {hw_config['device_name']}")
            print(f"   🔧 Batch size {hw_config['batch_size']}")
    
    def display_menu(self):
        """Affiche le menu principal"""
        print("\n📋 Actions disponibles:")
        print("   1. Calculer statistiques historiques")
        print("   2. Fine-tuner le modèle TimesFM") 
        print("   3. Générer 1 prédiction")
        print("   4. Série de N prédictions")
        print("   5. Afficher statut détaillé")
        print("   6. Quitter")
        
    def run_stats_calculation(self):
        """Exécute le calcul des statistiques"""
        stats = self.stats_analyzer.calculate_frequencies()
        if stats:
            loading_animation("Sauvegarde...", 1.0)
            saved = self.stats_analyzer.save_stats()
            if saved:
                print("✅ Statistiques calculées et sauvegardées")
                self.stats_analyzer.display_summary()
            else:
                print("❌ Erreur sauvegarde")
        else:
            print("❌ Erreur calcul statistiques")
    
    def run_finetuning(self):
        """Exécute le fine-tuning"""
        # Vérifier si des stats existent
        if not self.storage.load_stats():
            print("❌ Calculer d'abord les statistiques (option 1)")
            return
        
        success = self.fine_tuner.finetune_model(epochs=5)
        if success:
            loading_animation("Sauvegarde modèle...", 1.5)
            saved = self.fine_tuner.save_finetuned_model()
            if saved:
                print("✅ Fine-tuning terminé et sauvegardé")
            else:
                print("⚠️  Fine-tuning terminé mais erreur sauvegarde")
        else:
            print("❌ Erreur fine-tuning")
    
    def run_single_prediction(self):
        """Génère une prédiction simple"""
        loading_animation("Génération prédiction", 1.0)
        
        # Essayer avec TimesFM fine-tuné d'abord
        has_finetuned = self.storage.load_model() is not None
        
        if has_finetuned:
            pred = self.predictor.predict_with_timesfm(use_finetuned=True)
        else:
            print("⚠️  Aucun modèle fine-tuné - utilisation stats seules")
            pred = self.predictor.predict_single_combination()
        
        # Affichage de la prédiction
        print("\n🎲 PRÉDICTION GÉNÉRÉE:")
        print(f"   Boules: {pred['boules']}")
        print(f"   Chance: {pred['chance']}")
        print(f"   Confiance: {pred['confidence']:.1%}")
        print(f"   Méthode: {pred['method']}")
    
    def run_prediction_series(self):
        """Génère une série de prédictions"""
        try:
            n_str = input("\n📊 Nombre de prédictions (défaut: 100): ").strip()
            n_predictions = int(n_str) if n_str else 100
            
            if n_predictions < 1:
                print("⚠️  Minimum 1 prédiction")
                n_predictions = 1
            
            has_finetuned = self.storage.load_model() is not None
            results = self.validator.generate_prediction_series(
                n_predictions=n_predictions, 
                use_timesfm=has_finetuned
            )
            
            # Affichage des résultats
            self.validator.display_series_summary(results)
            
        except ValueError:
            print("❌ Nombre invalide")
        except KeyboardInterrupt:
            print("\n⏹️  Opération annulée")
    
    def run_detailed_status(self):
        """Affiche un statut détaillé"""
        print("\n📊 STATUT DÉTAILLÉ")
        print("-" * 30)
        
        # Statut des données
        status = self.storage.get_status()
        print(f"Stats calculées: {status.get('stats', '❌')}")
        print(f"Modèle fine-tuné: {status.get('finetuning', '❌')}")
        
        # Infos sur les stats si disponibles
        stats_data = self.stats_analyzer.load_stats()
        if stats_data and stats_data.get('metadata'):
            meta = stats_data['metadata']
            print(f"\n📈 Données historiques:")
            print(f"   {meta['total_draws']} tirages analysés")
            print(f"   Période: {meta['date_range']['first']} → {meta['date_range']['last']}")
        
        # Infos sur le modèle
        model_info = self.fine_tuner.get_model_info()
        print(f"\n🤖 Modèle TimesFM:")
        print(f"   Modèle base: {'✅' if model_info.get('model_loaded') else '❌'}")
        print(f"   Type: {model_info.get('model_type', 'unknown')}")
        print(f"   Fine-tuné: {'✅' if model_info.get('is_finetuned') else '❌'}")
        print(f"   Sauvegardé: {'✅' if model_info.get('has_saved_model') else '❌'}")
    
    def run(self):
        """Lance l'interface CLI"""
        self.display_header()
        
        while True:
            self.display_status()
            self.display_menu()
            
            try:
                choice = input("\n🎯 Votre choix (1-6): ").strip()
                
                if choice == '1':
                    self.run_stats_calculation()
                elif choice == '2':
                    self.run_finetuning()
                elif choice == '3':
                    self.run_single_prediction()
                elif choice == '4':
                    self.run_prediction_series()
                elif choice == '5':
                    self.run_detailed_status()
                elif choice == '6':
                    print("\n👋 Au revoir!")
                    break
                else:
                    print("❌ Choix invalide")
                
                input("\nAppuyez sur Entrée pour continuer...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir!")
                break
            except Exception as e:
                print(f"\n❌ Erreur: {e}")
                input("Appuyez sur Entrée pour continuer...")


def main():
    """Point d'entrée principal"""
    try:
        cli = LotoV2CLI()
        cli.run()
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()