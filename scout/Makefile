.PHONY: lint types arch schema test check ci plugin

lint:
	uv run ruff check . && uv run ruff format --check .
types:
	uv run pyright
arch:
	uv run lint-imports
schema:
	uv run python scripts/gen_schema.py --check
test:
	uv run pytest -q
check: lint types arch schema test
check-all: lint types arch schema
	uv run pytest -q -m 'not live and not eval and not spike'
plugin:
	@if [ -d integrations/claude-code ]; then claude plugin validate ./integrations/claude-code; else echo "integrations/claude-code absent, skipping"; fi
ci: check plugin
