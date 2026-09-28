# TODO: Add a test target and .github/workflows/ci.yml for locked installs,
# unittest discovery, lint and types. Fix learn/ issues while keeping
# intentionally unfinished exercise tests separate from application tests.
.PHONY: install run debug clean lint
ARGS ?= --help
install:
	uv sync
run:
	uv run python -m src $(ARGS)
debug:
	uv run python -m pdb -m src $(ARGS)
clean:
	find src $(wildcard tests) -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache
lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
