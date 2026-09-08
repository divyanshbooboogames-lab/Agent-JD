"""Human-facing interface.

Presentation only. It imports the same `Agent` the REST API calls, so both
paths run identical retrieval and identical reasoning -- the UI just renders
the evidence, caveats and tool trace that the API returns as JSON.

Run:
    streamlit run src/agentjd/ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run path/to/app.py` without installing the package first.
SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st  # noqa: E402

from agentjd.agent.core import Agent, available_options  # noqa: E402
from agentjd.agent.schemas import AgentRequest  # noqa: E402
from agentjd.settings import get_settings  # noqa: E402

st.set_page_config(page_title="Agent JD -- sector intelligence",
                   page_icon="*", layout="wide")

SAMPLE_QUESTIONS = [
    "Is this sector a good place to be putting money to work right now?",
    "Which companies in this sector look like attractive buyout targets "
    "based on the data you have?",
    "Which of these companies would fit a long-term core holding versus a "
    "name I should avoid?",
    "Walk me through the margin profile of the companies in your data -- "
    "who's improving and who's under pressure?",
    "If I had to pick one company here to take private, which would it be "
    "and what's the operational thesis?",
    "What's the most recent headcount or hiring signal you have for UPS?",
    "What do you think about Ferrari?",
]


@st.cache_resource
def get_agent() -> Agent:
    return Agent()


@st.cache_data
def get_options() -> dict:
    return available_options()


def main() -> None:
    settings = get_settings()
    options = get_options()
    personas = {p["label"]: p for p in options["personas"]}
    sectors = {s["label"]: s for s in options["sectors"]}

    st.title("Agent JD")
    st.caption("One agent, three analyst personas, four sectors. All data "
               "reaches the agent through MCP tools.")

    with st.sidebar:
        st.header("Configuration")
        persona_label = st.selectbox("Persona", list(personas))
        sector_label = st.selectbox("Sector", list(sectors))
        persona, sector = personas[persona_label], sectors[sector_label]

        st.markdown("**Lens**")
        st.caption(persona["lens"])
        st.markdown("**Sector**")
        st.caption(sector["description"])

        st.divider()
        provider = settings.effective_provider()
        if provider == "anthropic":
            st.success(f"LLM: {settings.model}")
        else:
            st.warning(
                "Running the deterministic provider (no API key found). "
                "Retrieval, persona weighting and the MCP boundary all work; "
                "the answer is composed from templates rather than written by "
                "a model. Set ANTHROPIC_API_KEY for full answers.")
        if not settings.resolved_db_path.exists():
            st.error("No database. Run: python -m agentjd.ingest.build_db")

    st.subheader("Ask a question")
    sample = st.selectbox("Start from a sample question", ["(write my own)"]
                          + SAMPLE_QUESTIONS)
    default = "" if sample == "(write my own)" else sample
    query = st.text_area("Question", value=default, height=90,
                         placeholder="Ask about this sector...")

    if st.button("Ask", type="primary", disabled=not query.strip()):
        with st.spinner(f"Thinking as a {persona['label']}..."):
            try:
                response = get_agent().ask_sync(AgentRequest(
                    query=query.strip(), persona=persona["id"],
                    sector=sector["id"]))
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {exc}")
                return

        st.markdown(f"### {response.persona_label} on {response.sector_label}")
        st.markdown(response.answer)

        cols = st.columns(4)
        cols[0].metric("Confidence", response.confidence)
        cols[1].metric("Companies cited", len(response.companies_referenced))
        cols[2].metric("Tool calls", len(response.tool_calls))
        cols[3].metric("Elapsed", f"{response.elapsed_ms / 1000:.1f}s")

        if response.out_of_scope:
            st.warning("Not in this database, so not analysed: "
                       + ", ".join(response.out_of_scope))

        if response.evidence:
            st.markdown("#### Evidence")
            st.dataframe(
                [
                    {
                        "Company": e.company or "-",
                        "Metric": e.metric,
                        "Value": e.value,
                        "Unit": e.unit or "-",
                        "Context": e.context or "-",
                        "Derived": "yes" if e.is_derived else "no",
                    }
                    for e in response.evidence
                ],
                use_container_width=True, hide_index=True)

        if response.caveats:
            with st.expander(f"Data caveats ({len(response.caveats)})"):
                for c in response.caveats:
                    st.markdown(f"- {c}")

        with st.expander(f"MCP tool trace ({len(response.tool_calls)} calls)"):
            st.caption("Every fact above arrived through one of these calls. "
                       "The agent holds no database connection of its own.")
            st.dataframe(
                [
                    {
                        "Tool": c.tool,
                        "Arguments": str(c.arguments),
                        "OK": c.ok,
                        "ms": c.latency_ms,
                        "Result": c.result_summary,
                    }
                    for c in response.tool_calls
                ],
                use_container_width=True, hide_index=True)

        with st.expander("Raw JSON (identical to POST /v1/ask)"):
            st.json(response.model_dump())

        if response.data_sources:
            st.caption("Sources: " + " | ".join(response.data_sources))


main()
