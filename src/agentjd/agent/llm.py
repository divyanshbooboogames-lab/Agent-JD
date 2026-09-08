"""LLM providers.

Two implementations sit behind one interface:

* `AnthropicProvider` -- runs a manual agentic loop against the Messages API.
  A manual loop rather than the SDK's tool runner because the tools are
  discovered over MCP at runtime, and because every round trip is recorded for
  the response's audit trail.

* `DeterministicProvider` -- answers with no LLM at all. It still goes through
  the same MCP tools and the same persona weighting, and composes the result
  from the persona's own answer structure. It exists so the system is runnable
  and reviewable without credentials, and so tests can assert on retrieval
  behaviour without paying for or depending on model output.

Both return the same `AgentAnswer`, and the response always names which one ran.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from ..config import Persona, Sector
from ..formatting import fmt_value, pct_rank
from ..settings import Settings
from .mcp_client import McpToolbox
from .prompts import build_system_prompt, build_user_prompt
from .schemas import AgentAnswer, Evidence, answer_json_schema


class LLMProvider(Protocol):
    name: str
    model: str | None

    async def answer(self, *, query: str, persona: Persona, sector: Sector,
                     toolbox: McpToolbox, max_rounds: int) -> AgentAnswer:
        ...


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider:
    """Manual agentic loop over the Messages API."""

    def __init__(self, settings: Settings) -> None:
        from anthropic import AsyncAnthropic  # imported lazily: optional at runtime

        self.settings = settings
        self.name = "anthropic"
        self.model = settings.model
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key or None)
        # Server-side fallbacks route around a safety refusal instead of
        # returning nothing. Disabled automatically if the account cannot use
        # the beta -- see `_create`.
        self._use_fallbacks = True

    async def _create(self, **kwargs: Any) -> Any:
        """Send one request, degrading gracefully if the fallback beta is off."""
        if self._use_fallbacks:
            try:
                return await self._client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                if not _is_beta_rejection(exc):
                    raise
                self._use_fallbacks = False
        return await self._client.messages.create(**kwargs)

    async def answer(self, *, query: str, persona: Persona, sector: Sector,
                     toolbox: McpToolbox, max_rounds: int) -> AgentAnswer:
        tools = await toolbox.discover()
        system = build_system_prompt(persona, sector, toolbox.tool_names)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": build_user_prompt(query, persona, sector)}
        ]

        request: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "tools": tools,
            "output_config": {
                "effort": self.settings.effort,
                "format": {"type": "json_schema", "schema": answer_json_schema()},
            },
        }

        response = None
        for _ in range(max_rounds):
            response = await self._create(messages=messages, **request)

            if response.stop_reason == "refusal":
                return _refusal_answer(response, persona)

            if response.stop_reason == "pause_turn":
                # A server-side tool ran long; resend to let it continue.
                messages.append({"role": "assistant", "content": response.content})
                continue

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in tool_uses:
                # Tool inputs are parsed JSON from the SDK; never string-matched.
                payload, ok = await toolbox.call(block.name, dict(block.input or {}))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": payload,
                    **({"is_error": True} if not ok else {}),
                })
            messages.append({"role": "user", "content": results})
        else:
            # Budget exhausted with tools still pending: ask for the answer now
            # rather than returning a truncated tool transcript.
            messages.append({
                "role": "user",
                "content": ("You have used the full tool budget. Answer now "
                            "from what you already retrieved, and record the "
                            "gap in caveats."),
            })
            response = await self._create(messages=messages, **request)

        return _parse_answer(response, persona)


def _is_beta_rejection(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(s in text for s in ("beta", "fallbacks", "unexpected keyword"))


def _refusal_answer(response: Any, persona: Persona) -> AgentAnswer:
    details = getattr(response, "stop_details", None)
    reason = getattr(details, "explanation", None) or "the request was declined"
    return AgentAnswer(
        answer=f"I can't answer this one: {reason}.",
        confidence="low",
        caveats=["The model declined this request; no analysis was produced."],
    )


def _parse_answer(response: Any, persona: Persona) -> AgentAnswer:
    """Read the structured JSON the model was constrained to produce."""
    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text.strip():
        return AgentAnswer(
            answer="The model returned no text. Retry the request.",
            confidence="low",
            caveats=["Empty model response."])
    try:
        return AgentAnswer.model_validate_json(text)
    except Exception:  # noqa: BLE001
        # output_config guarantees valid JSON, but never let a parse failure
        # lose an answer the user could still read.
        return AgentAnswer(
            answer=text,
            confidence="low",
            caveats=["Response did not match the structured schema; the raw "
                     "model text is shown and the structured fields are empty."])


# ---------------------------------------------------------------------------
# Deterministic
# ---------------------------------------------------------------------------

#: Rough detector for company mentions in a question: quoted strings, all-caps
#: ticker-like tokens, and capitalised multi-word names. It is a heuristic and
#: is described as one -- its only job is to decide what to look up, and
#: `find_company` makes the actual in/out-of-database call.
_MENTION_RE = re.compile(
    r'"([^"]{2,40})"'
    r"|\b([A-Z]{2,5})\b"
    r"|\b((?:[A-Z][a-z]+)(?:\s+(?:[A-Z][a-z]+|&|of|and)){0,3})\b"
)

#: Question shapes that should be answered from stored signals rather than
#: from the sector screen. The LLM provider decides this for itself by reading
#: the tool descriptions; the deterministic provider needs the rule spelled out.
_SIGNAL_QUERY_TERMS = (
    "headcount", "head count", "employee", "employees", "hiring", "hire",
    "workforce", "staff", "layoff", "layoffs", "attrition",
)

_STOPWORDS = {
    "I", "The", "A", "An", "Which", "What", "Who", "Where", "When", "Why",
    "How", "Is", "Are", "Do", "Does", "If", "Walk", "Give", "Tell", "Show",
    "Should", "Would", "Could", "This", "That", "These", "Those", "PE", "MF",
    "EBITDA", "TTM", "USD", "CEO", "CFO", "IPO", "ROI",
}


def _candidate_mentions(query: str) -> list[str]:
    out: list[str] = []
    for match in _MENTION_RE.finditer(query):
        token = next((g for g in match.groups() if g), "").strip(" .,?!")
        if not token or token in _STOPWORDS or len(token) < 2:
            continue
        if token not in out:
            out.append(token)
    return out[:6]


class DeterministicProvider:
    """Composes an answer from MCP tool output with no model in the loop."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.name = "deterministic"
        self.model = None

    async def _answer_signals(self, query: str, persona: Persona,
                              sector: Sector, toolbox: McpToolbox,
                              tickers: list[str],
                              out_of_scope: list[str]) -> AgentAnswer:
        """Answer a workforce question strictly from stored signals."""
        lines: list[str] = []
        evidence: list[Evidence] = []
        caveats: list[str] = []
        held = 0

        unique_tickers = list(dict.fromkeys(tickers))[:4]
        for ticker in unique_tickers:
            payload = await toolbox.call_json(
                "get_company_signals", {"ticker": ticker, "signal_type": "headcount"})
            signals = payload.get("signals") or []
            name = payload.get("name", ticker)
            if not signals:
                lines.append(
                    f"{name} ({ticker}): no headcount or hiring signal is held "
                    f"in this database.")
                if payload.get("guidance"):
                    caveats.append(payload["guidance"])
                continue
            held += 1
            for sig in signals[:3]:
                lines.append(
                    f"{name} ({ticker}): {sig.get('value_text') or sig.get('value_num')} "
                    f"(as of {sig.get('as_of') or 'unknown date'}, "
                    f"source: {sig.get('source_name') or 'unrecorded'}).")
                evidence.append(Evidence(
                    company=ticker, metric="headcount",
                    value=sig.get("value_num"), unit="count",
                    context=sig.get("value_text"), as_of=sig.get("as_of"),
                    source=sig.get("source_name")))

        if held == 0:
            lines.append("")
            lines.append(
                "Nothing is being estimated here. The loaded ingest adapter "
                "carries market and financial data but no workforce data, so "
                "the honest answer is that this signal is absent. Running the "
                "EDGAR adapter populates headcount for filers that tag "
                "dei:EntityNumberOfEmployees.")

        caveats.append("Answer composed by the deterministic provider; no "
                       "language model reasoned over this evidence.")
        if out_of_scope:
            caveats.append(
                "These were named in the question but are not in the database: "
                + ", ".join(out_of_scope))

        return AgentAnswer(
            answer="\n".join(lines),
            companies_referenced=unique_tickers,
            evidence=evidence, caveats=caveats[:6],
            out_of_scope=out_of_scope,
            confidence="high" if held else "low")

    async def answer(self, *, query: str, persona: Persona, sector: Sector,
                     toolbox: McpToolbox, max_rounds: int) -> AgentAnswer:
        out_of_scope: list[str] = []
        resolved: list[str] = []
        for mention in _candidate_mentions(query):
            found = await toolbox.call_json("find_company", {"query": mention})
            if found.get("in_database") is False:
                out_of_scope.append(mention)
            else:
                resolved.extend(m["ticker"] for m in found.get("matches", [])[:2])

        # A question about workforce is a signals question, not a screening
        # question. Answer it from what is stored -- including when nothing is.
        if any(term in query.lower() for term in _SIGNAL_QUERY_TERMS) and resolved:
            return await self._answer_signals(
                query, persona, sector, toolbox, resolved, out_of_scope)

        screen = await toolbox.call_json(
            "screen_sector",
            {"sector": sector.id, "persona": persona.id, "limit": 5})
        benchmarks = await toolbox.call_json(
            "get_sector_benchmarks", {"sector": sector.id})
        coverage = await toolbox.call_json(
            "describe_data_coverage", {"sector": sector.id})

        results = screen.get("results", [])
        if not results:
            return AgentAnswer(
                answer=(f"No companies are loaded for {sector.label}, so there "
                        f"is nothing to analyse. Rebuild the database for this "
                        f"sector before asking again."),
                confidence="low",
                caveats=["Empty sector universe."],
                out_of_scope=out_of_scope)

        medians = {b["metric_code"]: b for b in benchmarks.get("benchmarks", [])}
        evidence: list[Evidence] = []
        lines: list[str] = []

        weight_summary = ", ".join(
            f"{metric} {spec['weight']:.0%} ({spec['direction']})"
            for metric, spec in
            screen.get("ranking_basis", {}).get("weights", {}).items())

        lines.append(
            f"Reading {sector.label} as a {persona.label}, across the "
            f"{screen.get('universe_size', 0)} companies loaded for this "
            f"sector. The ranking below comes from this persona's own "
            f"weighting -- {weight_summary} -- not a generic screen.")
        lines.append("")
        lines.append(persona.answer_sections[0])

        for record in results:
            drivers = record.get("contributions", [])[:3]
            driver_text = "; ".join(
                f"{d['metric']} {fmt_value(d['value'], d.get('unit'))} "
                f"({pct_rank(d['percentile_in_sector'])})"
                for d in drivers)
            lines.append(
                f"  {record['rank']}. {record['name']} ({record['ticker']}) "
                f"- score {record['score']:.2f}. Driven by: {driver_text}.")
            for d in drivers:
                evidence.append(Evidence(
                    company=record["ticker"], metric=d["metric"],
                    value=d["value"], unit=d.get("unit"),
                    context=(f"{pct_rank(d['percentile_in_sector'])} "
                             f"in {sector.id}; weight {d['weight']} "
                             f"({d['direction']} is better)"),
                    is_derived=bool(d.get("is_derived")),
                    as_of=d.get("period_end")))

        lines.append("")
        lines.append("Sector anchors")
        for code in persona.priority_metrics[:4]:
            bench = medians.get(code)
            if bench:
                lines.append(
                    f"  - median {code}: "
                    f"{fmt_value(bench['median'], bench.get('unit'))} "
                    f"(n={bench['n']}, IQR "
                    f"{fmt_value(bench.get('p25'), bench.get('unit'))} to "
                    f"{fmt_value(bench.get('p75'), bench.get('unit'))})")
                evidence.append(Evidence(
                    metric=f"{code}_sector_median", value=bench["median"],
                    unit=bench.get("unit"),
                    context=f"median across {bench['n']} {sector.id} companies",
                    is_derived=True))

        lines.append("")
        lines.append(
            "This is a deterministic screen, not a written analyst view: it "
            "reports the ranking and the numbers behind it without "
            "interpretation. Set AGENTJD_LLM_PROVIDER=anthropic with an API "
            "key for reasoned commentary in the persona's voice.")

        caveats = [f["detail"] for f in coverage.get("data_quality_findings", [])
                   if f["severity"] in ("warn", "error")][:5]
        caveats.append("Answer composed by the deterministic provider; no "
                       "language model reasoned over this evidence.")
        if out_of_scope:
            caveats.append(
                "These were named in the question but are not in the database: "
                + ", ".join(out_of_scope))

        return AgentAnswer(
            answer="\n".join(lines),
            companies_referenced=[r["ticker"] for r in results],
            evidence=evidence[:20],
            caveats=caveats,
            out_of_scope=out_of_scope,
            confidence="low")


def build_provider(settings: Settings) -> LLMProvider:
    provider = settings.effective_provider()
    if provider == "anthropic":
        return AnthropicProvider(settings)
    return DeterministicProvider(settings)
