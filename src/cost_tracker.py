import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

# Fallback Naira rates per 1M tokens if /v1/models does not list a model
FALLBACK_RATES_NGN_PER_M: dict[str, dict[str, float]] = {
    "gemini-embedding-2": {"input": 297.0, "output": 0.0, "cache_read": 297.0},
    "gpt-6-luna": {"input": 1500.0, "output": 6000.0, "cache_read": 375.0},
    "gpt-5.6-luna": {"input": 1200.0, "output": 4800.0, "cache_read": 300.0},
    "claude-sonnet-5-5": {"input": 3000.0, "output": 15000.0, "cache_read": 300.0},
    "gemini-3.8-flash": {"input": 400.0, "output": 1600.0, "cache_read": 100.0},
    "gemini-3.5-flash-lite": {"input": 150.0, "output": 600.0, "cache_read": 37.5},
}


class NairaCostTracker:
    """Computes and logs per-request NGN costs for chat and embedding calls."""

    def __init__(self, client: OpenAI, log_path: str = "logs/request_costs.jsonl") -> None:
        self.client = client
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._rates: dict[str, dict[str, float]] = dict(FALLBACK_RATES_NGN_PER_M)
        self._refresh_rates()

    def _refresh_rates(self) -> None:
        try:
            models_resp = self.client.models.list()
            for m in getattr(models_resp, "data", []):
                model_id = getattr(m, "id", None)
                extra = getattr(m, "model_extra", None) or {}
                wekoya = getattr(m, "wekoya", None) or extra.get("wekoya") or {}
                cpm = wekoya.get("cost_per_million") if isinstance(wekoya, dict) else None
                if model_id and isinstance(cpm, dict):
                    inp = float(cpm.get("input", 0.0) or 0.0)
                    out = float(cpm.get("output", inp) or 0.0)
                    cache_read = float(cpm.get("cache_read", inp) or inp)
                    self._rates[model_id] = {
                        "input": inp,
                        "output": out,
                        "cache_read": cache_read,
                    }
        except Exception:
            # Keep fallback NGN rates if model listing is restricted or offline
            pass

    def get_rate(self, model: str) -> dict[str, float]:
        if model in self._rates:
            return self._rates[model]
        for key, rate in self._rates.items():
            if model.startswith(key):
                return rate
        return {"input": 1000.0, "output": 4000.0, "cache_read": 250.0}

    def record_embedding(
        self,
        session_id: str,
        operation: str,
        model: str,
        tokens: int,
    ) -> dict[str, Any]:
        rate = self.get_rate(model)
        cost_ngn = round((max(0, tokens) / 1_000_000) * rate["input"], 6)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "operation": operation,
            "model": model,
            "prompt_tokens": tokens,
            "cached_tokens": 0,
            "completion_tokens": 0,
            "cost_ngn": cost_ngn,
            "currency": "NGN",
        }
        self._append(entry)
        return entry

    def record_chat(
        self,
        session_id: str,
        operation: str,
        model: str,
        usage: Any,
    ) -> dict[str, Any]:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        uncached_prompt_tokens = max(0, prompt_tokens - cached_tokens)

        rate = self.get_rate(model)
        cost_ngn = round(
            (uncached_prompt_tokens / 1_000_000) * rate["input"]
            + (cached_tokens / 1_000_000) * rate["cache_read"]
            + (completion_tokens / 1_000_000) * rate["output"],
            6,
        )

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "operation": operation,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "cost_ngn": cost_ngn,
            "currency": "NGN",
        }
        self._append(entry)
        return entry

    def _append(self, entry: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
