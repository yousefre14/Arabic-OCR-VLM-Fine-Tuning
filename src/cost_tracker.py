"""cost_tracker.py — Token usage and cost accounting."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CostTracker:
    """Accumulates token counts and computes running API cost.
    """

    price_per_1m_input: float
    price_per_1m_output: float
    prompt_tokens: int = field(default=0, init=False)
    completion_tokens: int = field(default=0, init=False)

    def update(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    @property
    def total_cost(self) -> float:
        cost_in = (self.prompt_tokens / 1_000_000) * self.price_per_1m_input
        cost_out = (self.completion_tokens / 1_000_000) * self.price_per_1m_output
        return cost_in + cost_out

    def log(self, iteration: int, every_n: int = 3) -> None:
        """Log cost every `every_n` iterations."""
        if iteration % every_n == 0:
            logger.info(
                "Iteration %d | prompt=%d  completion=%d  cost=$%.4f",
                iteration,
                self.prompt_tokens,
                self.completion_tokens,
                self.total_cost,
            )
