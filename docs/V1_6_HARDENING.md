# V1.6 Evidence & Evaluation Hardening

V1.6 intentionally does not add more Agents. It hardens the V1.5 supervisor
architecture around completion, degradation, temporal provenance and evaluation.

## 1. Task Contract and Completion Gate

The Supervisor step limit is now a cost budget, not a completion criterion.

Each user request is converted into a TaskContract containing required entities,
required dimensions and critical requirements. After each capability execution,
CompletionGate produces:

- completion_ratio
- completed_items
- missing_items
- critical_missing
- evidence_gaps

When the step budget is exhausted, the system returns PARTIAL instead of silently
presenting an incomplete search as complete.

## 2. Explicit response semantics

Conversation responses expose one of:

- COMPLETE
- PARTIAL
- REVIEW_REQUIRED
- DATA_UNAVAILABLE
- SYSTEM_ERROR

The response also carries missing_items, evidence_gaps, unavailable_sources and a
user_action_required field. This separates research uncertainty from infrastructure
failure and makes Human-in-the-loop continuation possible without pausing every Agent.

## 3. Temporal provenance

New user-uploaded documents record:

- publish_date_source
- publish_date_confidence
- publish_date_verified

User-supplied dates are unverified by default. During historical research,
explicitly-unverified documents fail closed and are excluded from RAG. They remain
stored and may still be used for current-date document Q&A.

Legacy indexed documents without provenance remain compatible; a migration/verification
job is still recommended before claiming strict PIT coverage over old knowledge.

## 4. Component-source fail closed

Sector component fetches now preserve diagnostics for empty responses, missing columns
and upstream exceptions. If any selected sector lacks verifiable component data,
Representative Pool returns no representatives with COMPONENT_DATA_UNAVAILABLE instead
of silently building a biased partial pool.

Pure Sector Discovery remains available, so data-quality failure does not collapse the
entire user experience.

## 5. Evidence and Hypothesis Ledgers

Context compression now allocates separate budgets:

- Evidence Ledger: FACT / CALCULATION
- Hypothesis Ledger: INFERENCE / CONDITIONAL

This keeps grounding conservative without erasing the reasoning/hypothesis channel
needed by Bull/Bear research.

## 6. Evaluation harness

V1.6 adds:

- RoutingEval: action/target accuracy, forbidden route violations, unnecessary deep
  research rate, steps and tool-call metrics.
- Claim-level GroundingEval: claim grounding rate, unsupported claim rate and source
  coverage using a deterministic heuristic.
- SystemEvalSummary: side-by-side schema for Single-Agent, Fixed Multi-Agent and
  Dynamic Supervisor comparisons.
- evaluation/datasets/routing_eval_v1.jsonl: initial routing cases.

No architecture win is claimed until the baselines run under the same model, tools,
cutoff and comparable budget.

## 7. MCP position

MCP remains optional and disabled by default. Local Python tools are the default
single-process path. MCP is treated as a deployment/integration adapter for remote or
third-party tools, not as evidence that the Agent architecture itself is better.

## 7. PIT-safe adaptive style weighting

The rule weights remain the default and fallback. If a precomputed walk-forward
Style IC history is supplied, the history must record both signal date and
available_date (the date when the forward-return label has matured). Historical
research only consumes rows with available_date < as_of_date, then computes trailing
EWMA style signals and shrinks the learned distribution back toward the Regime rule
prior. Missing availability provenance or insufficient history falls back to Rule.

This adds an adaptive mechanism but does not manufacture a valid training history;
the Style IC history itself still needs proper walk-forward construction and evaluation.

## Remaining work

The following are deliberately not claimed as solved:

- proving the adaptive Style IC history improves out-of-sample discovery;
- semantic/LLM judge for grounding beyond deterministic heuristics;
- automated trusted publication-date verification from first-party source APIs;
- measured Single-Agent vs Fixed Deep Research vs Dynamic Supervisor benchmark results.\n  The V1.7 harness exists, but real model/data runs are still required.
