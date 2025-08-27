# 🤝 Contributing to TimesPredict

Merci de votre intérêt pour contribuer à TimesPredict ! Ce guide vous aidera à démarrer.

## 📋 Code de conduite

En participant à ce projet, vous acceptez de respecter notre code de conduite. Soyez respectueux et bienveillant envers tous les contributeurs.

## 🎯 Comment contribuer

### 🐛 Signaler des bugs

1. Vérifiez que le bug n'a pas déjà été signalé dans les [Issues](../../issues)
2. Utilisez le template de bug report
3. Incluez autant de détails que possible

### ✨ Proposer des fonctionnalités

1. Vérifiez que la fonctionnalité n'a pas déjà été demandée
2. Utilisez le template de feature request
3. Expliquez clairement le problème résolu

### 💻 Contribuer au code

#### 🔧 Configuration de l'environnement

1. **Fork le repository**
   ```bash
   git clone https://github.com/VOTRE_USERNAME/TimesPredict.git
   cd TimesPredict
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou venv\Scripts\activate  # Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   pip install black isort flake8 pytest
   ```

4. **Tester l'installation**
   ```bash
   python install_and_test.py
   ```

#### 🌿 Workflow de développement

1. **Créer une branche**
   ```bash
   git checkout -b feature/ma-nouvelle-fonctionnalite
   # ou
   git checkout -b fix/correction-bug
   ```

2. **Développer votre fonctionnalité**
   - Écrivez du code propre et lisible
   - Ajoutez des docstrings aux fonctions importantes
   - Suivez les conventions Python (PEP 8)

3. **Tester vos changements**
   ```bash
   # Format du code
   black src/
   isort src/
   
   # Lint
   flake8 src/
   
   # Tests basiques
   python -m pytest tests/ # si des tests existent
   ```

4. **Commit et Push**
   ```bash
   git add .
   git commit -m "feat: ajouter nouvelle fonctionnalité X"
   git push origin feature/ma-nouvelle-fonctionnalite
   ```

5. **Créer une Pull Request**
   - Utilisez le template de PR
   - Décrivez clairement vos changements
   - Liez les issues pertinentes

## 📝 Standards de code

### 🎨 Style

- **Formatage** : Utilisez `black` avec les paramètres par défaut
- **Imports** : Organisez avec `isort`
- **Linting** : Suivez `flake8` (max line length: 88)
- **Naming** : 
  - Variables et fonctions : `snake_case`
  - Classes : `PascalCase`
  - Constantes : `UPPER_CASE`

### 📚 Documentation

- **Docstrings** : Format Google style
  ```python
  def ma_fonction(param1: str, param2: int) -> bool:
      """
      Description courte de la fonction.
      
      Args:
          param1: Description du paramètre 1
          param2: Description du paramètre 2
          
      Returns:
          Description de ce qui est retourné
          
      Raises:
          ValueError: Quand param2 est négatif
      """
  ```

- **Comments** : Expliquez le "pourquoi", pas le "quoi"
- **Type hints** : Utilisez les annotations de type Python

### 🧪 Tests

- Ajoutez des tests pour les nouvelles fonctionnalités
- Les tests doivent être dans le répertoire `tests/`
- Nommage : `test_nom_de_la_fonction.py`
- Utilisez `pytest` comme framework

## 📁 Structure du projet

```
TimesPredict/
├── src/timesfm_predict/          # Code source principal
│   ├── models/                   # Modèles et wrappers
│   ├── data/                     # Traitement des données
│   ├── utils/                    # Utilitaires
│   └── examples/                 # Scripts d'exemple
├── tests/                        # Tests unitaires
├── docs/                         # Documentation (future)
├── .github/                      # Configuration GitHub
└── data/                         # Données d'exemple
```

## 🔄 Types de contributions

### 🐛 Corrections de bugs
- Identifiez clairement le problème
- Incluez des tests de régression
- Message de commit : `fix: description courte`

### ✨ Nouvelles fonctionnalités
- Discutez d'abord via une issue
- Implémentez avec des tests
- Message de commit : `feat: description courte`

### 📚 Documentation
- Améliorez les docstrings
- Mettez à jour le README
- Message de commit : `docs: description courte`

### ⚡ Performance
- Benchmarkez avant/après
- Documentez les gains
- Message de commit : `perf: description courte`

## 🎯 Priorités actuelles

- [ ] Tests unitaires complets
- [ ] Interface graphique (Streamlit/Gradio)
- [ ] Support d'autres APIs météo
- [ ] Optimisation des performances
- [ ] Documentation détaillée

## ❓ Questions ?

- 💬 Ouvrez une [Discussion](../../discussions)
- 🐛 Créez une [Issue](../../issues) avec le label "question"
- 📧 Contactez les mainteneurs

## 🏆 Reconnaissance

Tous les contributeurs seront ajoutés au README et dans les releases notes. Merci de faire de TimesPredict un meilleur projet ! 🚀

---

**Note :** Ce guide évolue avec le projet. N'hésitez pas à suggérer des améliorations !