"""Rolling-horizon beam search used by the hybrid planner."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from config import DIRECTIONS

Cell = Tuple[int, int]
State = Tuple[int, int, int]
ACTIONS = ("forward", "turn_left", "turn_right", "backward")


@dataclass(order=True)
class BeamNode:
    cost: float
    state: State
    actions: Tuple[str, ...]


def simulate(state: State, action: str, static_grid) -> Tuple[State, bool]:
    r, c, direction = state
    if action == "turn_left":
        return (r, c, (direction - 1) % 4), True
    if action == "turn_right":
        return (r, c, (direction + 1) % 4), True
    dr, dc = DIRECTIONS[direction]
    sign = 1 if action == "forward" else -1
    nr, nc = r + sign * dr, c + sign * dc
    rows, cols = static_grid.shape
    valid = 0 <= nr < rows and 0 <= nc < cols and static_grid[nr, nc] != 1
    return ((nr, nc, direction) if valid else state), valid


class RecedingHorizonPlanner:
    def __init__(self, horizon: int = 7, beam_width: int = 72):
        self.horizon = horizon
        self.beam_width = beam_width

    @staticmethod
    def _corridor_distance(cell: Cell, corridor: Sequence[Cell]) -> int:
        if not corridor:
            return 0
        return min(abs(cell[0] - p[0]) + abs(cell[1] - p[1]) for p in corridor[:20])

    def plan(
        self,
        state: State,
        goal: Cell,
        static_grid,
        risk_maps: List[Dict[Cell, float]],
        corridor: Sequence[Cell],
        use_prediction: bool = True,
        action_prior: Dict[str, float] | None = None,
        visit_counts: Dict[Cell, int] | None = None,
    ) -> Tuple[str, float]:
        action_prior = action_prior or {}
        visit_counts = visit_counts or {}
        beam = [BeamNode(0.0, state, tuple())]

        for depth in range(1, self.horizon + 1):
            expanded: List[BeamNode] = []
            risk = risk_maps[min(depth, len(risk_maps) - 1)] if use_prediction else risk_maps[0]
            for node in beam:
                for action in ACTIONS:
                    nxt, valid = simulate(node.state, action, static_grid)
                    if not valid:
                        continue
                    r, c, _ = nxt
                    dynamic_risk = risk.get((r, c), 0.0)
                    progress = abs(r - goal[0]) + abs(c - goal[1])
                    corridor_cost = self._corridor_distance((r, c), corridor)
                    action_cost = 1.0
                    if action in ("turn_left", "turn_right"):
                        action_cost += 0.32
                    elif action == "backward":
                        action_cost += 0.45
                    if nxt[:2] == node.state[:2]:
                        action_cost += 0.28
                    prior_bonus = 0.18 * action_prior.get(action, 0.0)
                    step_cost = (
                        action_cost + 0.28 * progress + 0.30 * corridor_cost
                        + 4.5 * dynamic_risk + 0.75 * visit_counts.get((r, c), 0)
                        - prior_bonus
                    )
                    expanded.append(BeamNode(node.cost + step_cost, nxt, node.actions + (action,)))
            if not expanded:
                return "turn_left", float("inf")
            # Keep diverse states: the cheapest action sequence per state wins.
            best_by_state = {}
            for node in sorted(expanded):
                best_by_state.setdefault(node.state, node)
            beam = sorted(best_by_state.values())[:self.beam_width]

        best = min(beam, key=lambda n: n.cost)
        return best.actions[0], best.cost
