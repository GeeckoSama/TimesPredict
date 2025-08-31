"""
Storage Module - Persistance des données Loto V2
Gestion JSON pour stats et pickle pour modèles
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class LotoStorage:
    """Gestionnaire de persistance pour Loto V2"""
    
    def __init__(self, base_dir: str = "data_v2"):
        self.base_dir = Path(base_dir)
        self.stats_dir = self.base_dir / "stats"
        self.models_dir = self.base_dir / "models"
        
        # Créer dossiers si inexistants
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_stats(self, stats_data: Dict[str, Any], name: str = "frequencies") -> bool:
        """Sauvegarde statistiques en JSON"""
        try:
            stats_with_metadata = {
                "data": stats_data,
                "timestamp": datetime.now().isoformat(),
                "version": "v2"
            }
            
            filepath = self.stats_dir / f"{name}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(stats_with_metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ Erreur sauvegarde stats: {e}")
            return False
    
    def load_stats(self, name: str = "frequencies") -> Optional[Dict[str, Any]]:
        """Charge statistiques depuis JSON"""
        try:
            filepath = self.stats_dir / f"{name}.json"
            if not filepath.exists():
                return None
                
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            return loaded.get("data", loaded)  # Support ancien format
        except Exception as e:
            print(f"❌ Erreur chargement stats: {e}")
            return None
    
    def save_model(self, model_data: Any, name: str = "finetuned_model") -> bool:
        """Sauvegarde modèle avec pickle"""
        try:
            filepath = self.models_dir / f"{name}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump({
                    "model": model_data,
                    "timestamp": datetime.now().isoformat(),
                    "version": "v2"
                }, f)
            return True
        except Exception as e:
            print(f"❌ Erreur sauvegarde modèle: {e}")
            return False
    
    def load_model(self, name: str = "finetuned_model") -> Optional[Any]:
        """Charge modèle depuis pickle"""
        try:
            filepath = self.models_dir / f"{name}.pkl"
            if not filepath.exists():
                return None
                
            with open(filepath, 'rb') as f:
                loaded = pickle.load(f)
            return loaded.get("model", loaded)  # Support ancien format
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            return None
    
    def get_status(self) -> Dict[str, str]:
        """Retourne le statut des opérations"""
        stats_exists = (self.stats_dir / "frequencies.json").exists()
        model_exists = (self.models_dir / "finetuned_model.pkl").exists()
        
        status = {}
        
        if stats_exists:
            try:
                with open(self.stats_dir / "frequencies.json", 'r') as f:
                    data = json.load(f)
                timestamp = data.get("timestamp", "Unknown")
                status["stats"] = f"✅ {timestamp[:19]}"
            except:
                status["stats"] = "✅ (pas de date)"
        else:
            status["stats"] = "❌ Non calculé"
            
        if model_exists:
            try:
                with open(self.models_dir / "finetuned_model.pkl", 'rb') as f:
                    data = pickle.load(f)
                timestamp = data.get("timestamp", "Unknown")
                status["finetuning"] = f"✅ {timestamp[:19]}"
            except:
                status["finetuning"] = "✅ (pas de date)"
        else:
            status["finetuning"] = "❌ Non fait"
            
        return status
    
    def clear_all(self) -> bool:
        """Supprime toutes les données sauvegardées"""
        try:
            for file in self.stats_dir.glob("*.json"):
                file.unlink()
            for file in self.models_dir.glob("*.pkl"):
                file.unlink()
            return True
        except Exception as e:
            print(f"❌ Erreur nettoyage: {e}")
            return False