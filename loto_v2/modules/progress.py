"""
Progress Module - Barres de progression pour Loto V2
Interface moderne avec barres de progression pour toutes les opérations
"""

import sys
import time
from typing import Optional


class UnifiedProgressBar:
    """Barre de progression unifiée élégante : [████████░░░░] étape/max - Action en cours"""
    
    def __init__(self, total_steps: int, operation_name: str = "Opération", width: int = 40):
        self.total_steps = total_steps
        self.current_step = 0
        self.operation_name = operation_name
        self.current_action = ""
        self.width = width
        self.start_time = time.time()
        
    def set_step(self, step: int, action: str = ""):
        """Met à jour l'étape courante et l'action"""
        self.current_step = min(step, self.total_steps)
        self.current_action = action
        self._display()
        
    def next_step(self, action: str = ""):
        """Passe à l'étape suivante"""
        self.set_step(self.current_step + 1, action)
        
    def update_action(self, action: str):
        """Met à jour seulement l'action courante"""
        self.current_action = action
        self._display()
        
    def _display(self):
        """Affiche la barre unifiée"""
        if self.total_steps == 0:
            return
            
        # Calcul du pourcentage
        percent = self.current_step / self.total_steps
        filled_width = int(self.width * percent)
        
        # Création de la barre élégante
        bar = "█" * filled_width + "░" * (self.width - filled_width)
        
        # Format : [████████░░░░] étape/max - Action en cours
        step_info = f"{self.current_step}/{self.total_steps}"
        action_text = f" - {self.current_action}" if self.current_action else ""
        
        line = f"\r{self.operation_name} [{bar}] {step_info}{action_text}"
        
        # Limiter la longueur de la ligne pour éviter les débordements
        max_line_length = 120
        if len(line) > max_line_length:
            # Tronquer l'action si nécessaire
            available_chars = max_line_length - len(line) + len(action_text)
            if available_chars > 10:
                truncated_action = self.current_action[:available_chars-3] + "..."
                line = f"\r{self.operation_name} [{bar}] {step_info} - {truncated_action}"
            else:
                line = f"\r{self.operation_name} [{bar}] {step_info}"
        
        sys.stdout.write(line)
        sys.stdout.flush()
        
        if self.current_step == self.total_steps:
            print()  # Nouvelle ligne à la fin
            
    def finish(self, final_action: str = "Terminé"):
        """Force la fin avec message final"""
        self.set_step(self.total_steps, final_action)


class ProgressBar:
    """Barre de progression simple pour terminal (compatibilité)"""
    
    def __init__(self, total: int, description: str = "", width: int = 40):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
        
    def update(self, increment: int = 1):
        """Met à jour la progression"""
        self.current = min(self.current + increment, self.total)
        self._display()
        
    def set_description(self, description: str):
        """Change la description"""
        self.description = description
        self._display()
        
    def _display(self):
        """Affiche la barre de progression"""
        if self.total == 0:
            return
            
        # Calcul du pourcentage
        percent = self.current / self.total
        filled_width = int(self.width * percent)
        
        # Création de la barre
        bar = "█" * filled_width + "░" * (self.width - filled_width)
        
        # Temps écoulé et estimation
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = f" ETA: {int(eta)}s" if eta > 1 else ""
        else:
            eta_str = ""
        
        # Affichage
        percent_str = f"{percent:.0%}"
        counter_str = f"{self.current}/{self.total}"
        
        line = f"\r{self.description} [{bar}] {percent_str} {counter_str}{eta_str}"
        sys.stdout.write(line)
        sys.stdout.flush()
        
        if self.current == self.total:
            print()  # Nouvelle ligne à la fin
            
    def finish(self):
        """Force la fin de la barre"""
        self.current = self.total
        self._display()


class TaskProgress:
    """Gestionnaire de tâches avec barre de progression"""
    
    @staticmethod
    def with_progress(func, total_steps: int, description: str, *args, **kwargs):
        """Exécute une fonction avec barre de progression"""
        progress = ProgressBar(total_steps, description)
        
        def update_callback(step: int = 1, desc: Optional[str] = None):
            if desc:
                progress.set_description(desc)
            progress.update(step)
        
        try:
            result = func(update_callback, *args, **kwargs)
            progress.finish()
            return result
        except Exception as e:
            progress.finish()
            raise e
    
    @staticmethod
    def simulate_work(callback, duration: float, steps: int = 10):
        """Simule du travail avec progression (pour tests)"""
        step_duration = duration / steps
        for i in range(steps):
            time.sleep(step_duration)
            callback(1)
        return True


# Fonctions utilitaires pour l'interface utilisateur
def loading_animation(text: str, duration: float = 2.0):
    """Animation de chargement simple"""
    chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end_time = time.time() + duration
    i = 0
    
    while time.time() < end_time:
        sys.stdout.write(f"\r{chars[i % len(chars)]} {text}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    
    sys.stdout.write(f"\r✅ {text}\n")
    sys.stdout.flush()


def simple_progress(current: int, total: int, prefix: str = "") -> str:
    """Génère une barre de progression simple en une ligne"""
    if total == 0:
        return f"{prefix}[████████████████████] 100%"
    
    percent = current / total
    filled = int(20 * percent)
    bar = "█" * filled + "░" * (20 - filled)
    
    return f"{prefix}[{bar}] {percent:.0%} ({current}/{total})"


def countdown(seconds: int, message: str = "Démarrage dans"):
    """Compte à rebours avec affichage"""
    for i in range(seconds, 0, -1):
        sys.stdout.write(f"\r{message} {i}s...")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write(f"\r{message} maintenant!   \n")
    sys.stdout.flush()