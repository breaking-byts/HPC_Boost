from __future__ import annotations
from typing import List

class GlobalFixedBaseline:
    """Hardcoded 4 most commonly cited HPC events."""
    def __init__(self):
        self.selected_events = ['Instruct', 'Core_cyc', 'L1D_Miss', 'BrMispred']

    def get_events(self) -> List[str]:
        return self.selected_events
