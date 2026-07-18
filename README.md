# prediction-market-debate

Two LLM analysts argue opposite sides of a live Polymarket question. A judge picks a winner. An eval harness scores all three of them and tells me what's actually broken.

Built with LangGraph, the OpenAI API (`gpt-4o-mini` throughout), and Polymarket's Python SDK. Started as ~200 lines of hand-rolled asyncio and raw Chat Completions calls; rewritten on LangGraph's `create_agent` once the manual tool-loop wiring stopped teaching me anything new. The manual version lives in git history.

## How it works

```
START → setup → bull ─┐
              → bear ─┴→ judge → END
```

- **setup** picks a market one of three ways: an explicit `market_id`, a keyword draw (shuffled walk through ~60 keywords spanning sports, crypto, politics, entertainment, tech, and world events), or a random page-1 draw as fallback. Keyword draws only accept active markets priced between 0.35 and 0.65 — the reason is below.
- **bull** and **bear** run in parallel. Each is a `create_agent` ReAct loop with two tools — `market_lookup` (live Polymarket prices and probabilities) and `search` (web search via the OpenAI Responses API) — and returns a structured `AnalystResult` through `response_format`.
- **judge** reads both cases plus the market question and returns a structured `Verdict`: BULL or BEAR, with reasoning.

## The eval harness (`eval.py`)

Running the debate once tells you nothing about whether it's any good. So:

1. `generate_testing_ids(n)` draws n unique markets through the keyword path
2. All n debates run concurrently with `asyncio.gather`
3. Each debate gets three structured evaluations from a separate scoring model:
   - bull's and bear's arguments against an analyst rubric — consistency, phrasing, evidence_use, persuasiveness, relevance, each 0–20
   - the verdict against a judge rubric — consistency, phrasing, identification, reasoning_traceability, decision_alignment
4. Scores are averaged per criterion per role; feedback strings are collected and synthesized into a plain-text report by a summarizer model

### Why the 0.35–0.65 price filter exists

Early eval runs kept drawing long-shot political markets ("Will [person] win in 2028?") priced at 1–2%. On those, bear gets a free win: "the market says 1%" is unbeatable, the judge went BEAR in 100% of debates, and the eval couldn't tell me anything about whether the judge actually weighs evidence. Restricting the test set to contested prices removed the structural freebie — in the first run after the filter, bull won 2 of 3.

The filter also catches a subtler dead-market class: markets whose outcome is already publicly known but which are still `active` at the API level. A market pinned at 0.999 sailed straight through `state.active` checks and handed one side a trivial win.

## What the eval actually caught

**The assigned-side contradiction failure.** On a market where the evidence overwhelmingly favored YES (Trump confirmed to attend the World Cup final, set to hand over the trophy), the bear agent wrote an entire paragraph of reasoning supporting YES — then ended with "Therefore, the evidence strongly supports a 'No' prediction." This is a known failure mode of forced-side debate systems: when one side is unarguable, the assigned-against model staples its required conclusion onto honest reasoning. The pipeline caught it three ways at once: bear's consistency score cratered to 13.7 (the lowest any criterion has scored in any run), the judge called out the contradiction in its verdict, and the evaluator's feedback named the exact defect. I didn't design for this failure mode. The rubric surfaced it anyway.

**The judge trusts its priors over tool evidence.** Both analysts' web searches confirmed an event had already happened. The judge dismissed the claim as "not possible given the future date" — overriding fresh search results with its own stale training knowledge — and still reached the right verdict for partly wrong reasons. My rubric scored that reasoning highly. That's a gap in the rubric, not a win.

**reasoning_traceability is a stable floor.** Across every run, the judge's lowest criterion is reasoning_traceability (~15/20): verdicts consistently under-cite the specific claims they're supposedly weighing. It's the most consistent finding in the data and the first target for a judge-prompt revision.

**The evidence_use gap.** On lopsided market sets, bear grounds its case in `market_lookup` data while bull argues from narrative — a 12.3 vs 15.7 evidence_use split in one run. Requiring tool use in the system prompts plus contested-market selection narrowed it.

## Running it

```bash
git clone https://github.com/sircheatsalot24/prediction-market-debate
cd prediction-market-debate
uv sync
```

`.env` in the project root:

```
OPENAI_API_KEY=your_key_here
```

Then:

```bash
uv run main.py
```

That runs the full harness: draw markets, run the debates concurrently, score everything, print per-criterion averages and a synthesized report.

Requires Python 3.12+ (prompt construction uses nested f-string syntax from PEP 701) — `uv sync` handles this automatically via the repo's `.python-version`. Note the Polymarket SDK is in beta — its docs lag the code.

**Cost note:** a single 3-market pass is roughly 40–60 LLM calls once you count each agent's tool-calling rounds, nine scoring calls, and the summary. All on `gpt-4o-mini`, so it's cheap, but scale the sample size with that in mind.

## Files

- `main.py` — entry point; runs the full eval harness
- `graph.py` — the LangGraph pipeline: market selection, tools, bull/bear/judge nodes, sync and async entry points
- `evaluate.py` — the harness: test-set generation, concurrent runs, rubrics, scoring, aggregation, summary

## Known limitations

- Scores across runs aren't comparable yet. Each run draws a fresh test set, so a score change confounds "my system changed" with "the markets changed." Freezing a set to disk is the next step.
- The judge rubric can't detect "right verdict, wrong epistemics" — see the priors-over-evidence finding above.
- The evaluator scores whether evidence is *present*, not whether it's *appropriate*. Both analysts once conflated annual and daily air-quality metrics and nothing flagged it.
- Order bias is untested: bull is always presented first to the judge. Swapping presentation order across runs is on the list.

## Roadmap

- [ ] Persist frozen eval sets + results to disk for real A/B testing of prompt changes
- [ ] Judge-prompt revision targeting the reasoning_traceability floor
- [ ] Order-bias experiment (swap bull/bear order in the judge prompt)
- [ ] A rubric criterion for how the judge handles evidence that contradicts its priors