# Agent JD

A single persona-configurable financial analyst agent. One agent implementation
answers as a **Mutual Fund Analyst**, an **Equity Analyst**, or a **PE Analyst**,
over any of four sectors, reachable from a Streamlit UI or a REST API, with all
data access going through an **MCP** tool boundary.

The design claim this repository is built to demonstrate: **the persona changes
what the agent retrieves, not just how it writes.** Ask the same question about
the same sector as three personas and you get three different sets of companies,
because the persona's weighting is applied inside the MCP screening tool, below
the language model.

```
Sector: Technology -- 64 companies, identical universe for all three

MF Analyst              Equity Analyst          PE Analyst
--------------------    --------------------    --------------------
1. MU     0.79          1. FSLR   0.86          1. HPQ    0.86
2. MSFT   0.78          2. MU     0.85          2. ACN    0.86
3. NVDA   0.77          3. FICO   0.83          3. CTSH   0.81
4. ORCL   0.76          4. VRSN   0.83          4. SMCI   0.81
5. AVGO   0.73          5. PTC    0.83          5. HPE    0.80

Names appearing in all three top-5: none
```

Reproduce it with `make demo`. No API key required.

---

## Quickstart

```bash
make install                 # venv + dependencies
cp .env.example .env         # add ANTHROPIC_API_KEY for full answers
make db                      # build the database (~15s, GitHub only)
make test                    # 68 tests
make demo                    # persona divergence, no key needed

make api                     # REST on http://127.0.0.1:8000  (/docs for OpenAPI)
make ui                      # Streamlit on http://localhost:8501
```

**Running without an API key.** Every entry point works with no credentials.
With no key the agent falls back to a `deterministic` provider that still goes
through the same MCP tools and the same persona weighting, and composes the
answer from the persona's own structure instead of writing it with a model.
Retrieval, ranking, evidence, caveats, out-of-scope detection and the tool trace
are all real; only the prose is templated. Every response names which provider
produced it, so this is never ambiguous.

---

## Architecture

```
   Streamlit UI ──┐
                  ├──> Agent (agent/core.py) ──> LLM provider ──> Claude
   REST API ──────┘         │                    (or deterministic)
                            │
                            │  MCP (JSON-RPC over stdio)
                            ▼
                    MCP server (mcp_server/server.py)   <-- process boundary
                            │
                            ▼
                    SQLite, opened read-only
```

Both interfaces call the same `Agent.ask` and serialise the same
`AgentResponse`. Neither adds analysis of its own, so "one agent behind two
interfaces" is structural rather than a convention someone has to maintain.

**The agent holds no database handle.** It has no SQLite import, no SQL and no
schema knowledge. It launches the MCP server as a subprocess, discovers the tool
surface with `list_tools`, and acts only through `call_tool`. Replacing SQLite
with a warehouse, or moving the server to another host over streamable HTTP,
would not touch the agent.

### Layout

| Path | What it is |
|---|---|
| `src/agentjd/config/*.yaml` | Persona and sector definitions -- the behavioural config |
| `src/agentjd/ingest/` | Two pluggable source adapters + build pipeline + QA checks |
| `src/agentjd/db/` | Schema, metric registry, read-only query layer |
| `src/agentjd/mcp_server/` | The MCP tool surface |
| `src/agentjd/agent/` | Agent core, MCP client, LLM providers, prompts, schemas |
| `src/agentjd/api/`, `src/agentjd/ui/` | The two interfaces |

---

## The three personas

A persona is a YAML block, not a prompt string. Each carries three things that
bite at different layers:

| Layer | Field | Effect |
|---|---|---|
| Retrieval | `screen_weights` | Which companies come back from the database |
| Payload | `priority_metrics` | Which fields are attached, and in what order |
| Reasoning | `lens`, `answer_sections`, `guardrails` | How the evidence is argued |

The first layer is the one that matters. Prompt-only personas produce the same
ranked list in three voices.

`screen_sector` percentile-ranks each metric **within the sector**, applies the
persona's weights and directions, and renormalises over the metrics each company
actually has -- so a company is scored on what it has rather than punished for
gaps in the source data. Percentile ranks rather than z-scores because these
distributions are heavily skewed; a handful of mega-caps would dominate any
mean/standard-deviation scheme.

**Two weights are deliberately inverted for PE**, and they are the reason the
rankings diverge rather than merely reorder:

- `market_cap` is a **positive** weight for the mutual fund analyst (liquidity
  and index-relevant position sizes) and a **negative** weight for PE (a smaller
  enterprise is financeable as a take-private).
- `ebitda_margin_gap_to_sector` scores a company **higher when its margin is
  below** the sector median. An equity analyst reads that as a quality problem;
  a buyout shop reads it as the operational improvement that funds the return.

So the same database row ranks near the top for PE and near the bottom for the
mutual fund analyst. That inversion is asserted directly in the tests
(`test_mutual_fund_prefers_the_large_compounder_and_pe_prefers_the_small_cheap_one`).

---

## Data: what it is and where it came from

Two ingestion adapters write into the same schema. Both are real; neither
invents numbers.

### 1. `public` (default) -- public S&P 500 datasets on GitHub

- [`s-and-p-500-companies`](https://github.com/datasets/s-and-p-500-companies) --
  ticker, name, GICS sector and sub-industry, HQ, CIK, founding year, index add
  date. Sourced from Wikipedia, refreshed daily. This drives **sector
  membership**.
- [`s-and-p-500-companies-financials`](https://github.com/datasets/s-and-p-500-companies-financials)
  -- a point-in-time market snapshot (price, P/E, dividend yield, EPS, 52-week
  range, market cap, EBITDA, P/S, P/B), sourced via Yahoo Finance.

Both are ODC-PDDL-1.0 (public domain dedication). This adapter needs nothing but
GitHub, so `make db` works in a locked-down environment and the committed sample
database is reproducible by anyone.

### 2. `edgar` -- SEC EDGAR XBRL company facts

`make db-edgar` pulls annual (10-K, FY) XBRL facts straight from
`data.sec.gov`: revenue, net income, gross and operating income, cash, debt,
operating cash flow, capex, and headcount where a filer tags
`dei:EntityNumberOfEmployees`. Restatements supersede originals by filing date;
each metric resolves through an ordered list of concept aliases, because
companies tag the same quantity under different US-GAAP concepts.

This is the authoritative source and adds what the snapshot cannot: real
balance-sheet leverage, free cash flow, multi-year revenue growth, and
headcount. **It was not used to build the committed database**, because
`data.sec.gov` is blocked by the egress policy of the environment this was
authored in — which is precisely why the adapter split exists. Run
`make db-edgar` anywhere with normal outbound access. SEC requires a real
contact string in `AGENTJD_SEC_USER_AGENT` and caps clients at 10 req/s; the
adapter enforces 8 req/s.

### Current committed database

163 companies, 2,890 metric values, 72 sector benchmark rows, 4 sectors
(tech 73, manufacturing 54, retail 22, logistics 14).

### Known data-quality caveats

These are recorded as **rows in `data_quality_findings`**, not prose here, so
the agent cites them at answer time and the API returns them as structured
`caveats`. `describe_data_coverage` exposes them. The build recomputes them
every run, so they cannot drift away from the data they describe.

The standing ones for the default adapter:

1. **The market snapshot is undated.** The latest index-add date among covered
   companies (currently `2024-09-23`) is a firm *lower* bound — a company cannot
   appear in a constituent list compiled before it joined. No upper bound is
   claimed: ticker renames (`MMC`→`MRSH`, `BK`→`BNY`) make long-standing members
   look absent, which would date the snapshot to the 1980s. Treat prices and
   multiples as point-in-time.
2. **Revenue is reconstructed** as `market_cap / price_to_sales`, so every
   margin derived from it inherits any inconsistency between those two upstream
   fields. Directionally sound for ranking companies against their own sector;
   not the company's reported margin. Every derived value is flagged
   `is_derived` and stores the arithmetic that produced it.
3. **`ev_to_ebitda_proxy` excludes net debt** under the default adapter, because
   no balance-sheet data is loaded. It therefore understates the entry multiple
   for indebted businesses — which matters most to the persona that leans on it
   hardest. Never presented as a true EV/EBITDA. The EDGAR adapter fixes this.
4. **No headcount under the default adapter.** The brief's headcount question
   therefore gets an honest "no signal held", with the tool naming what would
   populate it. This is the intended behaviour, not a gap that got papered over.
5. **Logistics is thin** (14 companies). Medians over a set that small are
   indicative, not statistically meaningful; the build flags it.

---

## Schema decisions

`src/agentjd/db/schema.sql`. Three choices worth defending:

**Facts are stored long (EAV), not one column per metric.** Two adapters with
very different coverage — EDGAR exposes ~40 XBRL concepts, the snapshot exposes
9 — load into the same table with no migration, and the same table carries a
time series once multiple periods are ingested. The cost is clumsier ad-hoc SQL,
paid off by the `v_company_latest_metrics` and `v_company_facts` views. A
`metrics` registry table keeps the EAV self-describing and stops adapters
inventing near-duplicate metric names.

**Every fact carries a `source_id`.** Provenance is not decoration here: the
agent must state where a number came from and how stale it is, so "which source,
retrieved when" has to be queryable per value, not per database.

**Derived values are stored and flagged, not recomputed at query time.** Each
carries `is_derived` and the `derivation` string that produced it, so a reviewer
can tell a reported figure from an inferred one — and so can the agent, which is
instructed to say which it is using.

Two supporting tables exist because a persona needs them:
`sector_benchmarks` (the mutual fund analyst is *defined* as benchmark-relative,
so the sector median must be a stored, citable number rather than something the
model estimates from a list it was shown) and `company_signals` (workforce data
is textual, irregular, and the thing the brief stress-tests for hallucination —
it needs to be independently absent).

---

## MCP design

Nine tools, in `src/agentjd/mcp_server/server.py`:

| Tool | Purpose |
|---|---|
| `list_sectors` | What is in scope at all |
| `list_companies` | The universe being reasoned over |
| `find_company` | **Scope check** — returns `in_database: false` explicitly |
| `get_company_profile` | Everything held on one company, with provenance |
| `get_company_signals` | Headcount/hiring, or an explicit "nothing held" |
| `screen_sector` | **Persona-weighted ranking** — the primary evidence source |
| `get_sector_benchmarks` | Medians and quartiles, to anchor comparative claims |
| `compare_companies` | Side-by-side, naming any ticker not in the database |
| `describe_data_coverage` | Sources, licences, ingest runs, open QA findings |

Design notes:

- **Tools return explanations, not just values.** `screen_sector` returns each
  company's metric values, their percentile inside the sector, and each metric's
  contribution to the score, plus the weighting rationale. The agent can explain
  a ranking instead of asserting it, and a reviewer can audit it.
- **Absence is a first-class result.** `find_company` and `get_company_signals`
  return structured "not held" payloads carrying explicit guidance not to fill
  the gap from prior knowledge. Honest scope-awareness is designed into the tool
  contract rather than left to prompt wording.
- **The boundary is a real process boundary.** The agent spawns
  `python -m agentjd.mcp_server.server` and speaks JSON-RPC over its stdio. Tests
  can pass an in-process server instead; the same `Client` and the same protocol
  handle both, so the tested path and the production path differ only in
  transport.
- **Read-only by construction.** The connection is opened with SQLite's
  `mode=ro` URI and every statement is parameterised, so no tool argument — even
  one a model was talked into producing — can write to the database. Asserted in
  `test_database_is_opened_read_only`.

I used a **manual agentic loop** rather than the SDK's tool runner, because the
tools are discovered over MCP at runtime and because every round trip is
recorded for the response's audit trail. The Anthropic SDK does ship MCP
conversion helpers (`anthropic.lib.tools.mcp`); the tradeoff was making the
protocol boundary explicit and auditable over writing less code.

---

## The two interfaces

Both return the same object. The API serialises it; the UI renders it.

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' -d '{
  "query": "Which companies look like attractive buyout targets?",
  "persona": "pe_analyst",
  "sector": "logistics"
}' | jq
```

```jsonc
{
  "answer": "...",
  "persona": "pe_analyst",
  "persona_label": "Private Equity Analyst",
  "sector": "logistics",
  "companies_referenced": ["UAL", "DAL", "FDX", "LUV", "UPS"],
  "evidence": [
    {
      "company": "UAL",
      "metric": "ev_to_ebitda_proxy",
      "value": 5.14,
      "unit": "multiple",
      "context": "96th pct in logistics; weight 0.3 (low is better)",
      "is_derived": true
    }
  ],
  "caveats": ["..."],
  "out_of_scope": [],
  "confidence": "low",
  "tool_calls": [
    {"tool": "screen_sector", "ok": true, "latency_ms": 71, "result_summary": "..."}
  ],
  "provider": "deterministic",
  "model": null,
  "data_sources": ["S&P 500 market snapshot (via Yahoo Finance) (ODC-PDDL-1.0)"],
  "elapsed_ms": 1176
}
```

`tool_calls`, `provider`, `model` and `elapsed_ms` are **observed by the
harness, not asserted by the model** — a model cannot claim it called a tool it
did not call.

Routes: `GET /health`, `GET /v1/options`, `POST /v1/ask`, `GET /docs`.

---

## Sample queries from the brief

| Query | Behaviour |
|---|---|
| Same question, three personas | Different ranked companies and different framing — `make demo` |
| PE + Logistics, buyout targets | Ranks on entry multiple, EBITDA scale, margin headroom, small size |
| MF + Retail, core holding vs avoid | Ranks on growth, quality and size, benchmarked to the sector median |
| Equity + Manufacturing, margin profile | Leads on `ebitda_margin`/`net_margin` percentiles vs the peer median |
| "Headcount for [company]?" | `get_company_signals` → honest "no signal held" under the default adapter; real values after `make db-edgar` |
| "What about [company not in dataset]?" | `find_company` returns `in_database: false`; the ticker lands in `out_of_scope` and is not analysed |
| API POST with persona + sector | Structured JSON above, consumable programmatically |

---

## LLM choice

**Claude Opus 5** (`claude-opus-5`) via the Anthropic Python SDK. Configure with
`ANTHROPIC_API_KEY`; `AGENTJD_MODEL` and `AGENTJD_EFFORT` override the model and
effort level.

Implementation notes: adaptive thinking (on by default for this model), a
`json_schema` output constraint derived from the same pydantic model the API
returns — so the response contract cannot drift between the two — and
server-side refusal fallbacks enabled by default.

Both of those last two are opportunistic. If the account or model rejects the
fallback beta or the structured output format, `_create` retries once without
that feature and remembers the downgrade, so an optional capability can never
cost the whole request. The downgrade requires the error to name the feature
*and* read as a rejection — auth failures, timeouts and rate limits propagate
untouched rather than being mistaken for a capability problem.

---

## Testing

```bash
make test     # 68 tests, no network, no API key
```

Tests build their own small fixture database rather than leaning on the
committed one, so they assert on known values and stay green when the upstream
data refreshes. The fixture is constructed so the personas *must* disagree: a
large, expensive, high-margin compounder and a small, cheap, low-margin operator.

Coverage worth knowing about: persona weights sum to 1 and only reference
metrics that exist in the registry; the MF/PE direction inversion; derivations
are arithmetically correct and refuse to emit values with bad denominators; the
MCP tool surface round-trips over a real client; read-only enforcement;
out-of-scope and missing-signal honesty paths; and the API's structured-response
contract.

**The agentic loop is tested against a scripted Messages API**
(`tests/fake_anthropic.py`). Without an API key that loop is the one part of
the system that cannot run end to end, so rather than ship it unexecuted the
fake supplies scripted responses while the MCP tools underneath stay real -- a
scripted tool call really does query the fixture database. The suite drives
single-round answers, multi-round and parallel tool use, budget exhaustion,
`pause_turn`, refusals, malformed and empty responses, failing and unknown
tools, and the optional-feature downgrades. It asserts on the outbound request
shape too: user/assistant role alternation, `tool_use`/`tool_result` pairing,
and that the forced final call disables tools.

Writing it found three real bugs, all now fixed:

1. The budget-exhaustion path appended a second consecutive user message, which
   the Messages API rejects outright -- so hitting the tool limit was a
   guaranteed 400 rather than a graceful "answer with what you have".
2. That same forced call still offered tools, letting the model spend its last
   turn on another tool call and return a response with no text block.
3. Errors raised inside the MCP session came back wrapped in an anyio
   `ExceptionGroup`, so an authentication failure surfaced over HTTP as
   "unhandled errors in a TaskGroup (1 sub-exception)" with the real cause
   invisible. `Agent.ask` now unwraps a lone leaf.

The structured-output schema is also strict-compatible now: pydantic emits
nested models without `additionalProperties: false` and with only genuinely
required fields listed, which risks a 400 or silently dropped fields.

---

## What I would do next, with more time

1. **Run the EDGAR adapter as the primary source.** It is written and its
   parsing is unit-tested, but the committed database was built from the GitHub
   snapshot because `data.sec.gov` was blocked where this was authored. That
   swap alone fixes three of the five standing caveats: real net debt (so a true
   EV/EBITDA instead of a proxy), free cash flow, and headcount.
2. **Time series instead of a snapshot.** The schema already carries
   `period_end` and `fiscal_period`; nothing populates multiple periods under
   the default adapter. Multi-year data turns "margin profile" from a level into
   a trajectory, which is what the equity persona actually wants — the brief's
   "who's improving and who's under pressure" question is currently answerable
   only as a cross-section.
3. **Evaluate the personas rather than asserting they differ.** Ranking
   divergence is tested; *answer quality* is not. I would build a small eval set
   of the brief's questions with rubrics per persona (does the PE answer state
   an entry multiple and an exit path? does the MF answer size the position?) and
   score it, so persona prompt changes can be measured instead of eyeballed.
4. **Cache the persona screen and pre-warm the prompt.** The system prompt and
   tool list are stable per persona/sector pair — natural `cache_control`
   breakpoints that would cut latency and cost on repeat questions.
5. **Qualitative context.** Everything here is numeric. Filing text, press
   releases and news would let the PE persona argue an operational thesis from
   evidence rather than from margin gaps alone — and would need a retrieval
   layer with the same provenance discipline.

### Honest limitations

- Sector membership follows GICS mechanically. `manufacturing` excludes
  "Electronic Manufacturing Services" (classified under tech) and `retail`
  excludes restaurants. Defensible, documented, and arguable.
- Scores are relative positions inside one sector and one database. They are a
  screen to focus attention, not a recommendation.
- The deterministic provider's company-mention detection is a regex heuristic.
  It decides what to look up; `find_company` makes the actual in/out-of-database
  call, so a miss costs a lookup, not a wrong claim.
- Nothing here is investment advice.
