"""One-step safety filter for planner commands."""

from typing import Dict, Sequence, Tuple

from .local_planner import ACTIONS, simulate

Cell = Tuple[int, int]
State = Tuple[int, int, int]


class SafetyShield:
    def __init__(self, risk_threshold: float = 0.45):
        self.risk_threshold = risk_threshold
        self.interventions = 0

    def reset(self):
        self.interventions = 0

    def filter(
        self,
        proposed: str,
        state: State,
        static_grid,
        next_risk: Dict[Cell, float],
        observed: Sequence[Cell],
    ) -> str:
        observed = set(observed)

        def score(action):
            nxt, valid = simulate(state, action, static_grid)
            if not valid:
                return float("inf")
            cell = nxt[:2]
            immediate = 1.0 if cell in observed else 0.0
            wait_penalty = 0.08 if cell == state[:2] else 0.0
            return max(immediate, next_risk.get(cell, 0.0)) + wait_penalty

        proposed_score = score(proposed)
        if proposed_score < self.risk_threshold:
            return proposed

        ranked = sorted(ACTIONS, key=lambda a: (score(a), a != proposed))
        fallback = ranked[0]
        if fallback != proposed:
            self.interventions += 1
        return fallback
