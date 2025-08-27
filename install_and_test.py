"""
Script d'installation et de test pour TimesPredict
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description=""):
    """Exécute une commande et affiche le résultat"""
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"Commande: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes max
        )
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
            
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        if result.returncode == 0:
            print(f"✅ {description} - Succès")
        else:
            print(f"❌ {description} - Échec (code: {result.returncode})")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - Timeout (> 10 minutes)")
        return False
    except Exception as e:
        print(f"💥 {description} - Erreur: {e}")
        return False


def check_python_version():
    """Vérifie la version Python"""
    print("Vérification de la version Python...")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 11:
        print("✅ Version Python compatible")
        return True
    else:
        print("❌ Python 3.11+ requis pour TimesFM")
        return False


def install_dependencies():
    """Installe les dépendances"""
    print("\nInstallation des dépendances...")
    
    commands = [
        ("pip install --upgrade pip", "Mise à jour de pip"),
        ("pip install -r requirements.txt", "Installation des dépendances")
    ]
    
    for command, desc in commands:
        success = run_command(command, desc)
        if not success:
            print(f"Échec de l'installation: {desc}")
            return False
            
    return True


def test_imports():
    """Teste les imports principaux"""
    print("\nTest des imports...")
    
    test_modules = [
        "numpy",
        "pandas", 
        "matplotlib",
        "timesfm",
        "torch"
    ]
    
    failed_imports = []
    
    for module in test_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0


def test_project_structure():
    """Vérifie la structure du projet"""
    print("\nVérification de la structure du projet...")
    
    required_files = [
        "src/timesfm_predict/__init__.py",
        "src/timesfm_predict/models/timesfm_wrapper.py",
        "src/timesfm_predict/data/sales_data.py", 
        "src/timesfm_predict/data/weather_data.py",
        "src/timesfm_predict/examples/basic_sales_prediction.py",
        "requirements.txt"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0


def run_basic_test():
    """Lance un test de base"""
    print("\nTest de base du projet...")
    
    test_code = '''
import sys
import os
sys.path.append("src")

try:
    from timesfm_predict.data.sales_data import SalesDataProcessor
    
    # Test création de données
    processor = SalesDataProcessor()
    data = processor.create_sample_data(periods=30)
    
    print(f"✅ Test données: {len(data)} lignes créées")
    
    # Test préparation pour TimesFM
    sales_array, metadata = processor.prepare_for_timesfm()
    print(f"✅ Test préparation: {len(sales_array)} points")
    
    print("✅ Tests de base réussis")
    
except Exception as e:
    print(f"❌ Erreur lors du test: {e}")
    sys.exit(1)
'''
    
    try:
        exec(test_code)
        return True
    except Exception as e:
        print(f"❌ Test échoué: {e}")
        return False


def main():
    """Fonction principale d'installation et test"""
    
    print("🚀 INSTALLATION ET TEST DE TIMESPREDICT 🚀")
    print("=" * 60)
    
    # Vérifications préliminaires
    if not check_python_version():
        print("\n❌ Version Python incompatible. Arrêt.")
        return False
    
    # Structure du projet
    if not test_project_structure():
        print("\n❌ Structure de projet incomplète.")
        return False
    
    # Installation
    if not install_dependencies():
        print("\n❌ Échec de l'installation des dépendances.")
        return False
    
    # Test des imports
    if not test_imports():
        print("\n⚠️ Certains modules ne s'importent pas correctement.")
        print("Le projet peut fonctionner partiellement.")
    
    # Test de base
    if not run_basic_test():
        print("\n❌ Tests de base échoués.")
        return False
    
    # Succès
    print("\n" + "=" * 60)
    print("🎉 INSTALLATION TERMINÉE AVEC SUCCÈS!")
    print("=" * 60)
    
    print("\n📋 ÉTAPES SUIVANTES:")
    print("1. Testez l'exemple de base:")
    print("   python src/timesfm_predict/examples/basic_sales_prediction.py")
    print()
    print("2. Testez l'exemple avec météo:")
    print("   python src/timesfm_predict/examples/sales_with_weather.py")
    print()
    print("3. Explorez le code dans src/timesfm_predict/")
    print()
    print("4. Pour utiliser l'API météo (optionnel):")
    print("   - Copiez .env.example vers .env")
    print("   - Ajoutez votre clé API OpenWeatherMap")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💥 Installation échouée. Vérifiez les erreurs ci-dessus.")
        sys.exit(1)