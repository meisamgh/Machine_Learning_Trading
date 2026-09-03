.PHONY: setup lint test validate

setup:
	python3 -m venv .venv
	.venv/bin/pip install -e '.[dev]'

lint:
	.venv/bin/ruff check src tests

test:
	.venv/bin/pytest

validate: lint test
	git diff --check
