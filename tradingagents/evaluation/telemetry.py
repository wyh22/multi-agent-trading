from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult


def _model_name(
    serialized: dict[str, Any] | None,
    invocation_params: dict[str, Any] | None,
) -> str:
    params = invocation_params or {}
    for key in ("model", "model_name", "model_id"):
        value = params.get(key)
        if value:
            return str(value)

    serialized = serialized or {}
    kwargs = serialized.get("kwargs", {}) if isinstance(serialized, dict) else {}
    if isinstance(kwargs, dict):
        for key in ("model", "model_name", "model_id"):
            value = kwargs.get(key)
            if value:
                return str(value)

    name = serialized.get("name") if isinstance(serialized, dict) else None
    return str(name or "unknown")


class EvaluationTelemetryHandler(BaseCallbackHandler):
    """Thread-safe telemetry for architecture benchmarks.

    Token usage is provider-dependent. A provider that does not expose LangChain
    usage_metadata will still contribute call counts and latency, while token
    fields remain lower bounds rather than fabricated estimates.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.by_model: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "llm_calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
            }
        )
        self._run_models: dict[str, str] = {}
        self._started_llm_runs: set[str] = set()
        self._started_tool_runs: set[str] = set()

    def _start(
        self,
        *,
        serialized: dict[str, Any] | None,
        run_id: Any,
        invocation_params: dict[str, Any] | None,
    ) -> None:
        model = _model_name(serialized, invocation_params)
        key = str(run_id or "")
        with self._lock:
            if key and key in self._started_llm_runs:
                return
            if key:
                self._started_llm_runs.add(key)
                self._run_models[key] = model
            self.llm_calls += 1
            self.by_model[model]["llm_calls"] += 1

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            serialized=serialized,
            run_id=run_id,
            invocation_params=kwargs.get("invocation_params"),
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            serialized=serialized,
            run_id=run_id,
            invocation_params=kwargs.get("invocation_params"),
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage):
                usage = getattr(message, "usage_metadata", None)

        if not usage:
            llm_output = getattr(response, "llm_output", None) or {}
            if isinstance(llm_output, dict):
                usage = (
                    llm_output.get("token_usage")
                    or llm_output.get("usage")
                    or llm_output.get("usage_metadata")
                )

        if not isinstance(usage, dict):
            return

        input_tokens = int(
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        output_tokens = int(
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        )
        key = str(run_id or "")
        with self._lock:
            self.tokens_in += input_tokens
            self.tokens_out += output_tokens
            model = self._run_models.pop(key, "unknown") if key else "unknown"
            self.by_model[model]["tokens_in"] += input_tokens
            self.by_model[model]["tokens_out"] += output_tokens

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id or "")
        with self._lock:
            if key and key in self._started_tool_runs:
                return
            if key:
                self._started_tool_runs.add(key)
            self.tool_calls += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "llm_calls": int(self.llm_calls),
                "tool_calls": int(self.tool_calls),
                "tokens_in": int(self.tokens_in),
                "tokens_out": int(self.tokens_out),
                "tokens_total": int(self.tokens_in + self.tokens_out),
                "by_model": {
                    model: dict(values)
                    for model, values in self.by_model.items()
                },
            }


def estimate_cost(
    telemetry: dict[str, Any],
    pricing: dict[str, dict[str, float]] | None,
) -> float | None:
    """Estimate cost from user-supplied per-million-token pricing.

    Pricing is intentionally external rather than hard-coded because provider
    prices change. Example:
      {"gpt-x": {"input_per_million": 1.0, "output_per_million": 4.0}}
    """

    if not pricing:
        return None
    total = 0.0
    matched = False
    for model, usage in (telemetry.get("by_model") or {}).items():
        rate = pricing.get(model)
        if rate is None:
            # Allow a single "*" fallback for aliases/proxy model names.
            rate = pricing.get("*")
        if not rate:
            continue
        matched = True
        total += (
            float(usage.get("tokens_in", 0))
            / 1_000_000.0
            * float(rate.get("input_per_million", 0.0))
        )
        total += (
            float(usage.get("tokens_out", 0))
            / 1_000_000.0
            * float(rate.get("output_per_million", 0.0))
        )
    return total if matched else None
