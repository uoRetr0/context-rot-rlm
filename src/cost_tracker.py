"""Token and dollar tracking with budget enforcement."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from src.config import settings

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (USD)
PRICING: dict[str, dict[str, float]] = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-embedding-001": {"input": 0.006, "output": 0.0},
}


class BudgetExceededError(Exception):
    pass


@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0


@dataclass
class CostTracker:
    max_dollars: float = field(default_factory=lambda: settings.max_dollars)
    warn_at: float = field(default_factory=lambda: settings.warn_at_dollars)
    usage: dict[str, ModelUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def total_cost(self) -> float:
        return sum(u.cost_usd for u in self.usage.values())

    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.usage.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.usage.values())

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record usage and return cost for this call."""
        pricing = PRICING.get(model, {"input": 0.10, "output": 0.40})
        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )

        with self._lock:
            if model not in self.usage:
                self.usage[model] = ModelUsage()
            u = self.usage[model]
            u.input_tokens += input_tokens
            u.output_tokens += output_tokens
            u.calls += 1
            u.cost_usd += cost

        total = self.total_cost
        if total >= self.warn_at:
            logger.warning("Budget warning: $%.2f / $%.2f spent", total, self.max_dollars)

        return cost

    def check_budget(self) -> None:
        if self.total_cost >= self.max_dollars:
            raise BudgetExceededError(
                f"Budget exceeded: ${self.total_cost:.2f} >= ${self.max_dollars:.2f}"
            )

    def summary(self) -> str:
        lines = ["=== Cost Summary ==="]
        for model, u in sorted(self.usage.items()):
            lines.append(
                f"  {model}: {u.calls} calls, "
                f"{u.input_tokens:,} in / {u.output_tokens:,} out, "
                f"${u.cost_usd:.4f}"
            )
        lines.append(f"  TOTAL: ${self.total_cost:.4f}")
        return "\n".join(lines)


# Global singleton
tracker = CostTracker()
