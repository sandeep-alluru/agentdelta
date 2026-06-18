.PHONY: all test smoke lint typecheck fmt clean install docs demo

all: lint typecheck test

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --tb=short

smoke:
	python smoke_test.py --verbose

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/agentdelta/

docs:
	mkdocs serve

docs-build:
	mkdocs build --strict

demo:
	@command -v vhs >/dev/null 2>&1 || { echo "vhs not found — install from https://github.com/charmbracelet/vhs"; exit 1; }
	vhs assets/demo.tape

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info site/
