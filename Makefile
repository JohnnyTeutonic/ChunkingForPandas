.PHONY: lint typecheck test run clean docs docs-clean docs-serve benchmark

# Python interpreter to use
PYTHON := python3
# Source directory
SRC_DIR := src
# Test directory
TEST_DIR := tests

DOCS_DIR := docs
# Install dependencies

make install:
	pip install -e .
	
install-dev:
	pip install -e ".[dev]"

install-docs:
	pip install -e ".[docs]"

benchmark:
	$(PYTHON) benchmarks/benchmark_chunking.py

# Generate documentation
docs:
	sphinx-build $(DOCS_DIR)/source $(DOCS_DIR)/build/html

docs-clean:
	$(MAKE) -C $(DOCS_DIR) clean

docs-serve:
	sphinx-autobuild $(DOCS_DIR)/source $(DOCS_DIR)/build/html

# Run ruff linter
lint:
	ruff check $(SRC_DIR)
	ruff format $(SRC_DIR)

# Run mypy type checker
typecheck:
	mypy $(SRC_DIR)

# Run pytest with coverage
test:
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

# Run the Gradio app
run-app:
	chunking-interface

# Clean up Python cache files and build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +

# Run all checks
check: lint typecheck test

# Default target
all: install check