# PowerCast Project Makefile
# Convenient commands for development, testing, and deployment

.PHONY: help install install-dev test test-coverage lint format clean build docs run-notebook train evaluate

# Default target
help:
	@echo "PowerCast Project - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install       Install the package for production use"
	@echo "  install-dev   Install the package for development (includes dev dependencies)"
	@echo ""
	@echo "Development:"
	@echo "  test          Run all tests"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  lint          Run code linting (flake8, mypy)"
	@echo "  format        Format code with black and isort"
	@echo "  clean         Clean build artifacts and cache files"
	@echo ""
	@echo "Building and Documentation:"
	@echo "  build         Build the package for distribution"
	@echo "  docs          Generate documentation"
	@echo ""
	@echo "Analysis and Training:"
	@echo "  run-notebook  Start Jupyter Lab with the analysis notebook"
	@echo "  train         Train all PowerCast models"
	@echo "  evaluate      Evaluate trained models"
	@echo ""

# Installation commands
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,notebook]"
	pre-commit install

# Testing commands
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=powercast --cov-report=html --cov-report=term-missing

# Code quality commands
lint:
	flake8 src/powercast tests/
	mypy src/powercast

format:
	black src/powercast tests/
	isort src/powercast tests/

# Cleaning commands
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Building commands
build: clean
	python -m build

docs:
	cd docs && make html

# Analysis and ML commands
run-notebook:
	jupyter lab notebooks/powercast_analysis.ipynb

train:
	python -m powercast.cli train_models

evaluate:
	python -m powercast.cli evaluate_models

# Docker commands (if needed in future)
docker-build:
	docker build -t powercast:latest .

docker-run:
	docker run -it --rm -v $(PWD):/workspace powercast:latest

# CI/CD simulation
ci: lint test test-coverage

# Full development setup
setup-dev: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make run-notebook' to start analysis"
	@echo "Run 'make train' to train models"
	@echo "Run 'make test' to run tests"
