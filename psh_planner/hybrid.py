"""Predictive Safe Hybrid (PSH) global-local navigation stack."""

from typing import Callable, Dict, Sequence, Tuple

import numpy as np

from d_star_lite import DStarLite
from .local_planner import RecedingHorizonPlanner
from .prediction import DynamicObstacleTracker
from .safety import SafetyShield

Cell = Tuple[int, int]
State = Tuple[int, int, int]


class PredictiveSafeHybridPlanner:
    """D* Lite corridor + motion prediction + rolling search + safety shield."""

    name = "PSH (full)"

    def __init__(
        self,
        horizon: int = 7,
        beam_width: int = 72,
        use_prediction: bool = True,
        use_shield: bool = True,
        action_prior: Callable[[State], Dict[str, float]] | None = None,
    ):
        self.horizon = horizon
        self.use_prediction = use_prediction
        self.use_shield = use_shield
        self.action_prior = action_prior
        self.tracker = DynamicObstacleTracker(horizon=horizon)
        self.local = RecedingHorizonPlanner(horizon=horizon, beam_width=beam_width)
        self.shield = SafetyShield()
        self.global_planner = None
        self.static_grid = None
        self.goal = None
        self.last_corridor = []
        self.visit_counts = {}

    @property
    def shield_interventions(self):
        return self.shield.interventions

    def reset(self, static_grid, start: Cell, goal: Cell, start_dir: int = 1):
        self.static_grid = np.array(static_grid, dtype=int).copy()
        self.goal = goal
        self.global_planner = DStarLite(self.static_grid, start, goal)
        self.last_corridor = self.global_planner.plan()
        self.tracker.reset()
        self.shield.reset()
        self.visit_counts = {start: 1}

    def act(self, state: State, observed_dynamic: Sequence[Cell]) -> str:
        self.visit_counts[state[:2]] = self.visit_counts.get(state[:2], 0) + 1
        if self.global_planner.start != state[:2]:
            self.global_planner.update_start(state[:2])
            self.last_corridor = self.global_planner.replan()
        if not self.last_corridor:
            self.last_corridor = [state[:2], self.goal]

        self.tracker.update(observed_dynamic)
        risk_maps = self.tracker.predict(self.static_grid.shape, self.static_grid)
        prior = self.action_prior(state) if self.action_prior else None
        action, _ = self.local.plan(
            state, self.goal, self.static_grid, risk_maps, self.last_corridor,
            use_prediction=self.use_prediction, action_prior=prior,
            visit_counts=self.visit_counts,
        )
        if self.use_shield:
            shield_risk = risk_maps[min(1, len(risk_maps) - 1)] if self.use_prediction else risk_maps[0]
            action = self.shield.filter(
                action, state, self.static_grid, shield_risk,
                observed_dynamic,
            )
        return action
