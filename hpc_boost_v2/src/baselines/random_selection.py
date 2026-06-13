from __future__ import annotations
import random
from typing import List

class RandomBaseline:
    """Random K-event selection baseline (control)."""
    def __init__(self, k: int = 4, seed: int = 42):
        self.k = k
        self.seed = seed

    def select_events(self, event_columns: List[str], seed: int | None = None) -> List[str]:
        rng = random.Random(seed if seed is not None else self.seed)
        return rng.sample(event_columns, self.k)
