#!/usr/bin/env python3
"""
Test de détection hardware universelle
Compatible : Mac M4 Pro (MPS), RTX 4080 (CUDA), CPU multi-cœurs
"""

import sys
import torch
import platform
from pathlib import Path

# Ajouter le chemin des modules
sys.path.append(str(Path(__file__).parent))

from modules.finetuning import LotoFineTuner


def test_comprehensive_hardware_detection():
    """Test complet de détection hardware"""
    print("=" * 70)
    print("🔍 DÉTECTION HARDWARE UNIVERSELLE")
    print("=" * 70)
    print()
    
    # Informations système de base
    print("📋 Informations système :")
    print(f"   OS : {platform.system()} {platform.release()}")
    print(f"   Architecture : {platform.machine()}")
    print(f"   Processeur : {platform.processor()}")
    print(f"   Python : {sys.version.split()[0]}")
    print(f"   PyTorch : {torch.__version__}")
    
    # Test des capacités GPU
    print(f"\n🔍 Test des capacités GPU :")
    
    # CUDA (NVIDIA)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"   🟢 CUDA disponible : {gpu_count} GPU(s)")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"      GPU {i}: {name} ({memory:.1f} GB VRAM)")
        print(f"   ⚡ Version CUDA : {torch.version.cuda}")
    else:
        print("   ❌ CUDA non disponible")
    
    # MPS (Apple Silicon)
    if hasattr(torch, 'mps'):
        if torch.mps.is_available():
            print("   🟢 MPS (Metal) disponible")
        else:
            print("   🟡 MPS existe mais non disponible")
    else:
        print("   ❌ MPS non disponible")
    
    # CPU
    cpu_count = torch.get_num_threads()
    print(f"   🟢 CPU : {cpu_count} threads PyTorch")
    
    print(f"\n" + "─" * 70)
    print("🚀 CONFIGURATION AUTOMATIQUE OPTIMALE")
    print("─" * 70)
    
    # Test du système de détection
    fine_tuner = LotoFineTuner()
    config = fine_tuner.backend_config
    
    print(f"\n🎯 Configuration détectée :")
    print(f"   Backend : {config['backend'].upper()}")
    print(f"   Device : {config['device'].upper()}")
    print(f"   Hardware : {config['device_name']}")
    print(f"   Batch size : {config['batch_size']}")
    print(f"   Type GPU : {config['gpu_type']}")
    if 'memory_gb' in config:
        print(f"   VRAM : {config['memory_gb']:.1f} GB")
    
    # Recommandations selon le hardware
    print(f"\n💡 Optimisations détectées :")
    if config['gpu_type'] == 'nvidia_cuda':
        memory_gb = config.get('memory_gb', 0)
        if memory_gb >= 16:
            print("   🚀 GPU haute performance détecté")
            print("   ✅ Batch size élevé pour maximum de vitesse")
            print("   ✅ Idéal pour gros volumes de prédictions")
        elif memory_gb >= 12:
            print("   🚀 GPU RTX 4080 class détecté")
            print("   ✅ Configuration optimisée pour performance")
            print("   ✅ Excellent pour fine-tuning")
        else:
            print("   🔧 GPU NVIDIA standard")
            print("   ✅ Configuration équilibrée")
    elif config['gpu_type'] == 'apple_mps':
        print("   🍎 Apple Silicon optimisé")
        print("   ✅ Metal Performance Shaders")
        print("   ✅ Efficacité énergétique maximale")
    else:
        print("   🖥️  CPU multi-threading optimisé")
        print("   ✅ Batch size adapté aux cœurs disponibles")
    
    return config


def test_performance_comparison():
    """Test de performance selon le hardware"""
    print(f"\n" + "=" * 70)
    print("⚡ ESTIMATION DES PERFORMANCES")
    print("=" * 70)
    
    config = test_comprehensive_hardware_detection()
    
    # Estimations selon le hardware
    if config['gpu_type'] == 'nvidia_cuda':
        memory_gb = config.get('memory_gb', 0)
        if memory_gb >= 16:
            speed = "🚀 Très rapide"
            fine_tuning_time = "~2-3 minutes"
            predictions_100 = "~10-20 secondes"
        elif memory_gb >= 12:
            speed = "⚡ Rapide"  
            fine_tuning_time = "~3-5 minutes"
            predictions_100 = "~20-30 secondes"
        else:
            speed = "🔧 Correct"
            fine_tuning_time = "~5-8 minutes"
            predictions_100 = "~30-60 secondes"
    elif config['gpu_type'] == 'apple_mps':
        speed = "🍎 Efficace"
        fine_tuning_time = "~4-6 minutes"
        predictions_100 = "~25-40 secondes"
    else:
        cpu_threads = torch.get_num_threads()
        if cpu_threads >= 16:
            speed = "🖥️  CPU rapide"
            fine_tuning_time = "~8-12 minutes"
            predictions_100 = "~1-2 minutes"
        else:
            speed = "🖥️  CPU standard"
            fine_tuning_time = "~12-20 minutes" 
            predictions_100 = "~2-4 minutes"
    
    print(f"\n📊 Estimations de performance :")
    print(f"   Vitesse générale : {speed}")
    print(f"   Fine-tuning (5 epochs) : {fine_tuning_time}")
    print(f"   100 prédictions : {predictions_100}")
    print(f"   Batch size optimal : {config['batch_size']}")
    
    print(f"\n🎯 Recommandations d'usage :")
    if config['batch_size'] >= 12:
        print("   ✅ Parfait pour gros volumes de prédictions")
        print("   ✅ Fine-tuning rapide recommandé")
    elif config['batch_size'] >= 8:
        print("   ✅ Bon équilibre performance/stabilité")
        print("   ✅ Idéal pour usage quotidien")
    else:
        print("   🔧 Configuration conservatrice")
        print("   💡 Réduire le nombre de prédictions si lent")


def main():
    """Test principal de détection hardware"""
    try:
        test_comprehensive_hardware_detection()
        test_performance_comparison()
        
        print(f"\n🎉 Détection hardware terminée !")
        print("   ✅ Configuration automatique optimale")
        print("   ✅ Compatible Mac M4 Pro + RTX 4080 + CPU")
        print("   ✅ Performance maximale selon votre hardware")
        
    except Exception as e:
        print(f"❌ Erreur lors de la détection : {e}")


if __name__ == "__main__":
    main()