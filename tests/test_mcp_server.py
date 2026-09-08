"""The MCP boundary itself.

These tests drive the server through a real `mcp.Client`, so tool discovery,
argument validation and result serialisation are all exercised over the
protocol rather than by calling the Python functions directly.
"""

from __future__ import annotations

import json

import pytest
from mcp import Client

from agentjd.db.store import connect
from agentjd.settings import Settings

EXPECTED_TOOLS = {
    "list_sectors", "list_companies", "find_company", "get_company_profile",
    "get_company_signals", "screen_sector", "get_sector_benchmarks",
    "compare_companies", "describe_data_coverage",
}


@pytest.fixture
def mcp_app(monkeypatch, fixture_db):
    """The real server module, pointed at the fixture database."""
    from agentjd.mcp_server import server as server_module

    settings = Settings(db_path=fixture_db)
    monkeypatch.setattr(server_module, "get_settings", lambda: settings)
    return server_module.mcp


def _payload(result):
    if getattr(result, "structured_content", None):
        return result.structured_content
    return json.loads(result.content[0].text)


async def test_all_tools_are_discoverable(mcp_app):
    async with Client(mcp_app) as client:
        tools = (await client.list_tools()).tools
        assert {t.name for t in tools} == EXPECTED_TOOLS


async def test_every_tool_is_described_for_the_model(mcp_app):
    """Tool descriptions are the agent's only instructions on when to call
    what, so an undescribed tool is a silent behavioural bug."""
    async with Client(mcp_app) as client:
        for tool in (await client.list_tools()).tools:
            assert tool.description and len(tool.description) > 40, tool.name


async def test_screen_sector_round_trips(mcp_app):
    async with Client(mcp_app) as client:
        result = await client.call_tool(
            "screen_sector",
            {"sector": "tech", "persona": "pe_analyst", "limit": 3})
        data = _payload(result)
        assert data["persona"] == "pe_analyst"
        assert len(data["results"]) == 3
        assert data["results"][0]["rank"] == 1


async def test_personas_differ_across_the_protocol(mcp_app):
    """The differentiation must survive the protocol boundary, not just exist
    inside the query layer."""
    async with Client(mcp_app) as client:
        mf = _payload(await client.call_tool(
            "screen_sector",
            {"sector": "tech", "persona": "mutual_fund_analyst", "limit": 4}))
        pe = _payload(await client.call_tool(
            "screen_sector",
            {"sector": "tech", "persona": "pe_analyst", "limit": 4}))
    assert [r["ticker"] for r in mf["results"]] != [r["ticker"] for r in pe["results"]]


async def test_out_of_scope_company_over_the_protocol(mcp_app):
    async with Client(mcp_app) as client:
        data = _payload(await client.call_tool("find_company", {"query": "Ferrari"}))
    assert data["in_database"] is False


async def test_database_is_opened_read_only(fixture_db):
    """The tool surface must be incapable of mutating the database, not merely
    disinclined to."""
    conn = connect(fixture_db, read_only=True)
    with pytest.raises(Exception) as excinfo:
        conn.execute("DELETE FROM companies")
    assert "readonly" in str(excinfo.value).lower()
    conn.close()
