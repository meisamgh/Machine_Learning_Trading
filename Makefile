.PHONY: setup test lint validate

setup:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check src tests

validate: lint test
