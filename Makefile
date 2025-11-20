

# Déclaration des variables
PYTHON=python
ENV_NAME=venv
REQUIREMENTS=requirements.txt
CONFIG=config/config.yaml

# ============================================================================
# PHONY TARGETS (non-file targets)
# ============================================================================
.PHONY: help all setup data train test deploy clean notebook format lint security

# ============================================================================
# HELP TARGET (Default)
# ============================================================================
help:
	@echo "Available targets:"
	@echo "  make all        - Exécuter le pipeline ML complet"
	@echo "  make setup      - Configuration de l'environnement"
	@echo "  make format     - Formatage automatique du code"
	@echo "  make lint       - Vérification de la qualité du code"
	@echo "  make security   - Analyse de sécurité du code"
	@echo "  make data       - Préparation des données"
	@echo "  make train      - Entraînement du modèle (avec config par défaut)"
	@echo "  make train-dev  - Entraînement rapide (config développement)"
	@echo "  make train-prod - Entraînement optimisé (config production)"
	@echo "  make test       - Exécution des tests"
	@echo "  make deploy     - Déploiement du modèle"
	@echo "  make clean      - Nettoyage des fichiers temporaires"
	@echo "  make notebook   - Démarrage du serveur Jupyter Notebook"

# ============================================================================
# 0. Pipeline complet (ALL)
# ============================================================================
all:
	@echo "========================================"
	@echo "EXECUTION DU PIPELINE ML COMPLET"
	@echo "========================================"
	@echo ""
	@echo "Étape 1/6: Formatage du code..."
	@$(MAKE) format
	@echo ""
	@echo "Étape 2/6: Vérification de la qualité..."
	@$(MAKE) lint
	@echo ""
	@echo "Étape 3/6: Analyse de sécurité..."
	@$(MAKE) security
	@echo ""
	@echo "Étape 4/6: Préparation des données..."
	@$(MAKE) data
	@echo ""
	@echo "Étape 5/6: Entraînement du modèle..."
	@$(MAKE) train
	@echo ""
	@echo "Étape 6/6: Déploiement..."
	@$(MAKE) deploy
	@echo ""
	@echo "========================================"
	@echo "PIPELINE COMPLET TERMINÉ AVEC SUCCÈS!"
	@echo "========================================"

# ============================================================================
# 1. Configuration de l'environnement
# ============================================================================
setup:
	@echo "Création de l'environnement virtuel et installation des dépendances..."
	@$(PYTHON) -m venv $(ENV_NAME)
	@.\$(ENV_NAME)\Scripts\activate && pip install -r $(REQUIREMENTS)

# ============================================================================
# 2. Qualité du code, formatage automatique du code, sécurité du code, etc
# ============================================================================
format:
	@echo "Formatage du code avec black et isort..."
	@$(PYTHON) -m black . --exclude "/(\.git|\.venv|venv|env|__pycache__|\.pytest_cache)/"
	@$(PYTHON) -m isort . --skip-gitignore

lint:
	@echo "Vérification de la qualité du code..."
	@$(PYTHON) -m flake8 . --exclude=.git,__pycache__,.pytest_cache,venv,.venv,env --max-line-length=100 --ignore=E203,W503
	@$(PYTHON) -m pylint *.py --disable=C0111,C0103,R0913,R0914

security:
	@echo "Analyse de sécurité du code..."
	@$(PYTHON) -m bandit -r . -f screen --skip B101,B601

# ============================================================================
# 3. Préparation des données
# ============================================================================
data:
	@echo "Préparation des données..."
	@$(PYTHON) main.py --prepare --config $(CONFIG)

# ============================================================================
# 4. Entraînement du modèle
# ============================================================================
train:
	@echo "Entraînement du modèle avec configuration par défaut..."
	@$(PYTHON) main.py --config $(CONFIG)

train-dev:
	@echo "Entraînement rapide (mode développement)..."
	@$(PYTHON) main.py --config config/config.dev.yaml

train-prod:
	@echo "Entraînement optimisé (mode production)..."
	@$(PYTHON) main.py --config config/config.prod.yaml

train-fast:
	@echo "Entraînement rapide sans optimisation..."
	@$(PYTHON) main.py --config $(CONFIG) --no-optimize

# ============================================================================
# 5. Tests unitaires
# ============================================================================
test:
	@echo "Exécution des tests..."
	@$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term

# ============================================================================
# 6. Déploiement du modèle
# ============================================================================
deploy:
	@echo "Déploiement du modèle..."
	@echo "Copie des fichiers du modèle vers le répertoire de production..."
	@$(PYTHON) -c "import shutil; import pathlib; pathlib.Path('production').mkdir(exist_ok=True); [shutil.copy(f, 'production/') for f in pathlib.Path('models').rglob('*') if f.is_file()]"
	@echo "Modèle déployé avec succès!"

# ============================================================================
# 7. Nettoyage
# ============================================================================
clean:
	@echo "Nettoyage des fichiers temporaires..."
	@$(PYTHON) -c "import pathlib; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	@$(PYTHON) -c "import pathlib; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"
	@$(PYTHON) -c "import pathlib; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.mypy_cache')]"
	@$(PYTHON) -c "import pathlib; import shutil; shutil.rmtree('htmlcov', ignore_errors=True)"
	@$(PYTHON) -c "import pathlib; import shutil; shutil.rmtree('$(ENV_NAME)', ignore_errors=True)"
	@$(PYTHON) -c "import pathlib; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('*.pyc')]"
	@$(PYTHON) -c "import pathlib; pathlib.Path('.coverage').unlink(missing_ok=True)"
	@echo "Nettoyage terminé!"

# ============================================================================
# 8. Démarrage du serveur Jupyter Notebook
# ============================================================================
.PHONY: notebook
notebook:
	@echo "Démarrage de Jupyter Notebook..."
	@.\$(ENV_NAME)\Scripts\activate && jupyter notebook

# ============================================================================
# END OF MAKEFILE
# ============================================================================
