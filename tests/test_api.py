"""The REST interface, including the brief's API-specific acceptance test."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentjd.agent.core import Agent
from agentjd.settings import Settings


@pytest.fixture
def client(monkeypatch, fixture_db):
    from agentjd.api import main as api_main
    from agentjd.mcp_server import server as server_module

    settings = Settings(db_path=fixture_db, llm_provider="deterministic")
    monkeypatch.setattr(server_module, "get_settings", lambda: settings)
    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(api_main, "_agent",
                        Agent(settings=settings, mcp_server=server_module.mcp))
    return TestClient(api_main.app)


def test_health_reports_configuration(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database_present"] is True
    assert set(body["personas"]) == {"mutual_fund_analyst", "equity_analyst",
                                     "pe_analyst"}


def test_options_drives_the_ui_selectors(client):
    body = client.get("/v1/options").json()
    assert len(body["personas"]) == 3
    assert len(body["sectors"]) == 4


def test_ask_returns_structure_a_machine_can_consume(client):
    """The brief's API test: POST persona + sector + question, and confirm the
    response carries more than a text blob."""
    response = client.post("/v1/ask", json={
        "query": "Walk me through the margin profile of these companies.",
        "persona": "equity_analyst",
        "sector": "logistics",
    })
    assert response.status_code == 200
    body = response.json()

    assert body["answer"]
    assert body["persona"] == "equity_analyst"
    assert body["sector"] == "logistics"
    assert body["confidence"] in {"high", "medium", "low"}
    assert isinstance(body["companies_referenced"], list)
    assert body["companies_referenced"]
    assert body["evidence"], "callers need the values behind the answer"
    assert body["tool_calls"], "callers need the retrieval trace"
    assert body["data_sources"]

    evidence = body["evidence"][0]
    assert set(evidence) >= {"metric", "value", "is_derived"}


def test_persona_changes_the_api_response_for_one_question(client):
    payload = {"query": "Where would you deploy capital here?", "sector": "tech"}
    mf = client.post("/v1/ask", json={**payload,
                                      "persona": "mutual_fund_analyst"}).json()
    pe = client.post("/v1/ask", json={**payload, "persona": "pe_analyst"}).json()
    assert mf["companies_referenced"] != pe["companies_referenced"]
    assert mf["answer"] != pe["answer"]


@pytest.mark.parametrize("payload,field", [
    ({"query": "x", "persona": "day_trader", "sector": "tech"}, "persona"),
    ({"query": "x", "persona": "pe_analyst", "sector": "biotech"}, "sector"),
])
def test_invalid_selectors_are_rejected_with_the_valid_set(client, payload, field):
    response = client.post("/v1/ask", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == f"unknown_{field}"
    assert detail["valid"]


def test_empty_query_is_rejected_by_validation(client):
    assert client.post("/v1/ask", json={
        "query": "", "persona": "pe_analyst", "sector": "tech"}).status_code == 422
