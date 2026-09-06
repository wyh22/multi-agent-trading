# V1.7 Benchmark & Walk-forward Evaluation

V1.7 is an experiment layer on top of the V1.6 reliability architecture. It does
not add more Agents. Its purpose is to make architecture claims measurable.

## 1. Three architecture variants

The end-to-end benchmark runs the same cases against:

1. single-agent-all-tools
   - one LLM
   - the complete atomic tool catalog
   - no Supervisor
   - no specialist-Agent routing

2. fixed-deep-research
   - always runs Market / News / Fundamentals
   - Bull / Bear
   - Portfolio Manager
   - Decision Auditor
   - then synthesizes an answer to the original user question

3. dynamic-supervisor
   - Task Contract
   - Conversation Supervisor
   - Tool / Specialist Agent / Skill / Deep Research routing
   - Completion Gate
   - explicit degradation status

The benchmark does not assume which architecture is better.

## 2. Architecture benchmark

Run:

    python scripts/run_agent_benchmark.py \
      --systems single-agent-all-tools,fixed-deep-research,dynamic-supervisor \
      --dataset evaluation/datasets/architecture_benchmark_v1.jsonl \
      --output results/benchmark_v1

For a cheap smoke test:

    python scripts/run_agent_benchmark.py \
      --systems dynamic-supervisor \
      --limit 1 \
      --output results/benchmark_smoke

Outputs:

- raw_runs.jsonl
- per_case.csv
- summary.csv
- BENCHMARK_REPORT.md

Metrics include:

- route accuracy
- unnecessary deep-research rate
- claim grounding rate
- unsupported claim rate
- evidence coverage
- independent completion ratio
- PIT date-parameter violations
- future explicit-date mentions
- LLM calls
- Tool calls
- provider-reported tokens
- wall-clock latency
- optional estimated cost

## 3. Cost accounting

Provider prices are deliberately not hard-coded because they change.

Pass a pricing JSON:

    python scripts/run_agent_benchmark.py \
      --pricing-json my_pricing.json

Example structure:

    {
      "your-model-name": {
        "input_per_million": 1.0,
        "output_per_million": 4.0
      }
    }

A "*" entry may be used as a fallback for proxy/alias model names.

If the provider does not expose token usage through LangChain usage metadata, the
benchmark reports the observed token count as a lower bound instead of inventing one.

## 4. Completion evaluation

Each benchmark case carries an explicit contract:

- required_entities
- required_dimensions
- critical_requirements

An independent Completion Gate evaluates all three systems from their evidence and
final answer. These judge calls are evaluation overhead and are excluded from each
system's telemetry.

The Supervisor's own Completion Gate remains part of the Supervisor system cost.

## 5. Grounding evaluation

Grounding is evaluated at the claim level against the evidence captured during the
run. The current evaluator is deterministic and intentionally conservative.

It is useful for regression and ablation, but it is not presented as a perfect
semantic judge. A future independent LLM judge can be added as a second metric rather
than silently replacing deterministic evaluation.

## 6. PIT-safe Style IC experiment

The adaptive sector-style path requires two steps.

### Step A: collect a style panel

Cheap weekly smoke test:

    python scripts/build_sector_style_panel.py \
      --start 2024-01-01 \
      --end 2026-08-31 \
      --freq W-FRI \
      --output evaluation/data/sector_style_panel.csv

Formal 20-trading-day experiment:

    python scripts/build_sector_style_panel.py \
      --start 2023-01-01 \
      --end 2026-08-31 \
      --freq B \
      --output evaluation/data/sector_style_panel_daily.csv

The collector uses the existing sector analyzer. Upstream failures are written to a
sidecar failures JSONL rather than silently disappearing.

### Step B: construct Style IC history

For daily sampling:

    python scripts/build_style_ic_history.py \
      --panel evaluation/data/sector_style_panel_daily.csv \
      --forward-periods 20 \
      --output evaluation/data/style_ic_history.csv

The output stores both:

- date: signal cross-section date
- available_date: date when the forward-return label has fully matured

Adaptive weights may only use rows with:

    available_date < as_of_date

This prevents a subtle look-ahead leak where a historical IC row exists by signal
date but its future-return label was not yet knowable.

## 7. Recommended experiment order

1. Run one-case Supervisor smoke test.
2. Run all benchmark cases for Dynamic Supervisor.
3. Run Single Agent with the identical model/tool configuration.
4. Run Fixed Deep Research.
5. Compare summary.csv.
6. Investigate individual failures in raw_runs.jsonl.
7. Only then write performance numbers into README or interview material.
8. Separately build the daily Style IC history and compare Rule vs Adaptive discovery.

## 8. Claims that are still not allowed

Until real runs exist, do not claim:

- Supervisor improves grounding by X%.
- Multi-Agent is more accurate than Single-Agent.
- Adaptive style weights improve return or IC.
- MCP reduces latency.
- RAG improves every research query.

The repository now provides the machinery to measure those claims; measurement results
must come from actual runs under controlled conditions.
