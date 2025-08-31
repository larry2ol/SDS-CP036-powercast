# PowerCast Project Makefile
# Deep Learning Time-Series Power Consumption Forecasting

.PHONY: help install install-dev setup clean test lint format docs train predict evaluate all

# Default target
help:
	@echo "PowerCast - Deep Learning Time-Series Forecasting"
	@echo "=================================================="
	@echo ""
	@echo "Available commands:"
	@echo "  install      Install package and dependencies"
	@echo "  install-dev  Install package in development mode with dev dependencies"
	@echo "  setup        Set up project environment and dependencies"
	@echo "  clean        Clean temporary files and cache"
	@echo "  test         Run tests"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black"
	@echo "  docs         Generate documentation"
	@echo "  train        Train all neural network models"
	@echo "  predict      Make predictions with trained models"
	@echo "  evaluate     Evaluate model performance"
	@echo "  notebook     Start Jupyter notebook server"
	@echo "  all          Run complete pipeline (clean, install, train, evaluate)"

# Installation commands
install:
	pip install -e .

install-dev:
	pip install -e .[dev]
	pip install -e .[docs]

setup:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	@echo "Setup complete! You can now run 'make train' to train models."

# Cleaning commands
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

# Development commands
test:
	pytest tests/ -v --cov=src/modules --cov-report=html --cov-report=term

lint:
	flake8 src/modules/ --max-line-length=88 --extend-ignore=E203,W503
	mypy src/modules/ --ignore-missing-imports

format:
	black src/modules/ notebooks/ --line-length=88
	isort src/modules/ notebooks/ --profile=black

# Documentation
docs:
	cd docs && sphinx-build -b html . _build/html
	@echo "Documentation generated in docs/_build/html/index.html"

# Model training and evaluation
train:
	python -c "import sys; sys.path.append('src'); from modules.week3_neural_networks import main; main()"

predict:
	python -c "import sys; sys.path.append('src'); from modules.utils import predict_cli; predict_cli()"

evaluate:
	python -c "import sys; sys.path.append('src'); from modules.week3_neural_networks import evaluate_all_models; evaluate_all_models()"

# Jupyter notebook
notebook:
	jupyter notebook notebooks/

# Complete pipeline
all: clean setup train evaluate
	@echo "Complete PowerCast pipeline executed successfully!"

# Docker commands (if Docker is used)
docker-build:
	docker build -t powercast:latest .

docker-run:
	docker run -it --rm -v $(PWD):/workspace powercast:latest

# Data validation
validate-data:
	python -c "import sys; sys.path.append('src'); from modules.utils import validate_data; validate_data('data/data.csv')"

# Model deployment preparation
deploy-prep:
	mkdir -p deployment/
	cp models/*.h5 deployment/
	cp config/model_config.yaml deployment/
	cp requirements.txt deployment/
	@echo "Deployment files prepared in deployment/ directory"

# Performance profiling
profile:
	python -c "import sys; sys.path.append('src'); import cProfile; cProfile.run('from modules.week3_neural_networks import main; main()', 'profile_output.prof')"
	@echo "Performance profile saved to profile_output.prof"

# Environment info
env-info:
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "Installed packages:"
	@pip list | grep -E "(tensorflow|keras|pandas|numpy|scikit-learn|matplotlib|seaborn)"

# Quick start guide
quickstart:
	@echo "PowerCast Quick Start Guide"
	@echo "=========================="
	@echo "1. Set up environment: make setup"
	@echo "2. Train models: make train"
	@echo "3. Evaluate models: make evaluate"
	@echo "4. Start notebook: make notebook"
	@echo "5. View documentation: make docs"
