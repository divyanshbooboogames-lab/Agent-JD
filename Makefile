.PHONY: help install db db-edgar api ui mcp demo test eval clean

PY ?= python
VENV := .venv
BIN := $(VENV)/bin

help:
	@echo "make install    create .venv and install dependencies"
	@echo "make db         build the database (GitHub sources; no API key needed)"
	@echo "make db-edgar   rebuild from SEC EDGAR filings (needs data.sec.gov)"
	@echo "make demo       show the same sector ranked by all three personas"
	@echo "make api        run the REST API on :8000"
	@echo "make ui         run the Streamlit UI on :8501"
	@echo "make mcp        run the MCP server standalone on stdio"
	@echo "make test       run the test suite"
	@echo "make eval       score the agent against the brief's questions"

install:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e .
	@echo "\nNow: cp .env.example .env && make db"

db:
	PYTHONPATH=src $(BIN)/python -m agentjd.ingest.build_db --fresh

db-edgar:
	PYTHONPATH=src $(BIN)/python -m agentjd.ingest.build_db --adapter edgar --fresh

demo:
	PYTHONPATH=src $(BIN)/python scripts/demo_personas.py --sector tech

api:
	PYTHONPATH=src $(BIN)/uvicorn agentjd.api.main:app --reload --port 8000

ui:
	PYTHONPATH=src $(BIN)/streamlit run src/agentjd/ui/streamlit_app.py

mcp:
	PYTHONPATH=src $(BIN)/python -m agentjd.mcp_server.server

test:
	PYTHONPATH=src $(BIN)/python -m pytest tests/ -q

eval:
	PYTHONPATH=src $(BIN)/python -m agentjd.evals.runner

clean:
	rm -rf $(VENV) data/sector_intel.db .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
