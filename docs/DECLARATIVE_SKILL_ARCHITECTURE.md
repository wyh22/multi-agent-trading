# Declarative Skill Architecture

## 1. Why this layer exists

The project already had executable task-level Skills before this change. They were
registered from `tradingagents/skills/manifests/*.yaml` and exposed to the
Conversation Supervisor through the shared `CapabilityRegistry`.

This update standardizes that existing design instead of adding a second router.

A Skill is a reusable research capability with a bounded execution contract. It is
not an additional autonomous Agent.

Runtime flow:

```text
User Query
   |
Conversation Supervisor
   |
Capability Registry
   |---- Atomic Tool
   |---- Specialist Agent
   \---- Declarative Skill
             |
             +---- Existing Workflow / Subgraph
             +---- Tool / RAG
             \---- bounded completion / audit semantics
```

## 2. Single source of truth

Each built-in Skill has two complementary artifacts:

1. `manifests/<skill>.yaml` — machine-readable runtime metadata.
2. `<skill>/SKILL.md` — reviewer-facing execution specification.

The YAML manifest remains the runtime source of truth. `SKILL.md` documents the
same contract for maintainers, code reviewers and interview/demo use.

The project deliberately does **not** maintain a second keyword-based
`registry.yaml + matcher`. Duplicating routing metadata would drift from the
actual executable capability names already used by `ConversationSupervisor`.

## 3. Runtime manifest fields

Current machine-readable metadata includes:

- `name`
- `description`
- `requires_ticker`
- `requires_audit`
- `allowed_agents`
- `when_to_use`
- `constraints`
- `completion`

`SkillSpec.routing_description()` renders the usage hints and completion contract
into the Supervisor capability catalog. This means the LLM router sees declarative
Skill semantics without hard-coding every use case into one large prompt.

## 4. Built-in Skills

### sector_discovery

Deterministic sector research prioritization and optional Representative Research
Pool construction. Numerical ranking remains Python-owned rather than LLM-owned.

### document_evidence_analysis

PIT-aware multi-query retrieval over the shared Qdrant knowledge layer. It is used
for long-document questions and targeted evidence repair.

### company_comparison

Bounded multi-company comparison. It requires at least two explicit tickers and
keeps evidence independent per company.

### deep_stock_research

High-cost full single-stock research workflow. Market / News / Fundamentals collect
evidence, Bull / Bear generate complementary hypotheses, Portfolio Manager
synthesizes, and Auditor checks evidence/PIT consistency with bounded repair.

## 5. Why not make every module an Agent?

The project intentionally separates:

- deterministic workflow: dates, PIT guards, ranking and numeric calculation;
- Tool: atomic external/data capability;
- Specialist Agent: a bounded LLM↔ToolNode research loop;
- reasoning node: Bull / Bear / Portfolio Manager / Auditor;
- Skill: a reusable task-level workflow;
- Conversation Supervisor: the single conversation-level routing decision entry.

This avoids inflating the system by calling every LLM-backed node an autonomous
Agent.

## 6. Packaging and validation

`pyproject.toml` packages both runtime YAML manifests and `SKILL.md` files.
Regression tests verify that every executable built-in Skill has a matching
`SKILL.md`, routing metadata is loaded, and no duplicate keyword registry is
introduced.

## 7. Resume/interview boundary

Safe description:

> The Conversation Supervisor dynamically routes between Tools, specialist research
> Agents and reusable declarative Skills. Sector discovery, document evidence
> analysis, company comparison and full deep research are packaged as bounded
> task-level Skills.

Avoid claiming that the four Skills are four autonomous Agents. Their execution is
implemented by existing workflows, specialist loops, deterministic code and RAG.
