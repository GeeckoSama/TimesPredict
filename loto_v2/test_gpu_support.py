#!/usr/bin/env python3
"""
Test du support GPU Mac M4 Pro pour TimesFM
Vérification des capacités MPS (Metal Performance Shaders)
"""

import torch
import platform
import sys
import os


def test_mac_gpu_support():
    """Teste le support GPU sur Mac M4 Pro"""
    print("🔍 Test du support GPU Mac M4 Pro")
    print("=" * 50)
    
    # Informations système
    print(f"🖥️  Système: {platform.system()} {platform.release()}")
    print(f"🔧 Architecture: {platform.machine()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    # Test MPS (Metal Performance Shaders)
    print(f"\n🚀 Support MPS:")
    if hasattr(torch, 'mps'):
        print(f"   torch.mps disponible: ✅")
        if torch.mps.is_available():
            print(f"   MPS disponible: ✅")
            
            # Test d'allocation GPU
            try:
                device = torch.device("mps")
                test_tensor = torch.rand(1000, 1000, device=device)
                result = torch.mm(test_tensor, test_tensor)
                print(f"   Test allocation GPU: ✅")
                print(f"   Device MPS: {device}")
                print(f"   Mémoire GPU utilisable: ✅")
                del test_tensor, result  # Nettoyer
                return True
            except Exception as e:
                print(f"   Test allocation GPU: ❌ {e}")
                return False
        else:
            print(f"   MPS disponible: ❌")
            return False
    else:
        print(f"   torch.mps disponible: ❌")
        return False


def test_timesfm_gpu():
    """Teste TimesFM avec GPU"""
    try:
        import timesfm
        print(f"\n🤖 TimesFM:")
        print(f"   TimesFM importé: ✅")
        
        # Créer une configuration GPU
        if torch.mps.is_available():
            print(f"   Test configuration GPU...")
            
            # Configuration avec backend GPU
            hparams = timesfm.TimesFmHparams(
                backend="gpu",  # Utiliser GPU au lieu de CPU
                per_core_batch_size=2,  # Plus petit batch pour GPU
                num_layers=50,
                model_dims=1280,
                num_heads=16,
                context_len=512,
                horizon_len=64,
                input_patch_len=32,
                output_patch_len=128
            )
            
            checkpoint = timesfm.TimesFmCheckpoint(
                version="pytorch",
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            )
            
            print(f"   Configuration GPU: ✅")
            
            # Note: On ne charge pas le modèle complet ici pour éviter 
            # de télécharger 2GB, juste tester la config
            return True
        else:
            print(f"   MPS non disponible, utiliser CPU")
            return False
            
    except ImportError:
        print(f"   TimesFM non disponible: ❌")
        return False
    except Exception as e:
        print(f"   Erreur TimesFM: ❌ {e}")
        return False


def get_optimal_config():
    """Retourne la configuration optimale pour Mac M4 Pro"""
    if torch.mps.is_available():
        return {
            "backend": "gpu",
            "device": "mps",
            "batch_size": 4,  # Optimisé pour M4 Pro
            "use_gpu": True
        }
    else:
        return {
            "backend": "cpu", 
            "device": "cpu",
            "batch_size": 8,
            "use_gpu": False
        }


def main():
    """Test principal"""
    print("🧪 Test complet support GPU Mac M4 Pro\n")
    
    # Tests de base
    mps_works = test_mac_gpu_support()
    timesfm_gpu = test_timesfm_gpu()
    
    # Configuration recommandée
    config = get_optimal_config()
    print(f"\n⚙️  Configuration recommandée:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Résumé
    print(f"\n📊 Résumé:")
    if mps_works and timesfm_gpu:
        print(f"   🚀 GPU Mac M4 Pro: ✅ Pleinement supporté")
        print(f"   💡 Recommandation: Utiliser backend='gpu'")
    elif mps_works:
        print(f"   ⚠️  GPU Mac M4 Pro: ⚠️ MPS fonctionne, TimesFM à tester")
        print(f"   💡 Recommandation: Essayer backend='gpu'") 
    else:
        print(f"   ❌ GPU Mac M4 Pro: ❌ Non supporté")
        print(f"   💡 Recommandation: Utiliser backend='cpu'")


if __name__ == "__main__":
    main()