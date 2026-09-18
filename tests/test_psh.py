import unittest

import numpy as np

from grid_env import DynamicGridEnv
from psh_planner.local_planner import simulate
from psh_planner.prediction import DynamicObstacleTracker
from psh_planner.safety import SafetyShield
from d_star_lite import DStarLite


class EnvironmentTests(unittest.TestCase):
    def test_dynamic_reset_is_reproducible(self):
        grid = np.zeros((8, 8), dtype=int)
        env = DynamicGridEnv(
            grid, (1, 1), (6, 6), moving_obstacles=[(3, 3)],
            obstacle_patterns=["random"], seed=17,
        )
        first = []
        env.reset()
        for _ in range(5):
            env.step("turn_left")
            first.append(tuple(env.moving_obstacles))
        second = []
        env.reset()
        for _ in range(5):
            env.step("turn_left")
            second.append(tuple(env.moving_obstacles))
        self.assertEqual(first, second)


class PlannerComponentTests(unittest.TestCase):
    def test_dstar_replans_around_new_obstacle(self):
        grid = np.zeros((7, 7), dtype=int)
        planner = DStarLite(grid, (3, 1), (3, 5))
        original = planner.plan()
        self.assertIn((3, 3), original)
        planner.set_obstacle(3, 3)
        replanned = planner.replan()
        self.assertNotIn((3, 3), replanned)
        self.assertEqual(replanned[-1], (3, 5))

    def test_tracker_projects_velocity(self):
        grid = np.zeros((10, 10), dtype=int)
        tracker = DynamicObstacleTracker(horizon=3)
        tracker.update([(4, 3)])
        tracker.update([(4, 4)])
        prediction = tracker.predict(grid.shape, grid)
        self.assertGreater(prediction[2].get((4, 6), 0), 0.5)

    def test_shield_rejects_predicted_collision(self):
        grid = np.zeros((7, 7), dtype=int)
        shield = SafetyShield(risk_threshold=0.5)
        action = shield.filter(
            "forward", (3, 3, 1), grid, {(3, 4): 0.9}, [],
        )
        self.assertNotEqual(action, "forward")
        self.assertEqual(shield.interventions, 1)

    def test_simulation_respects_static_obstacle(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[2, 3] = 1
        nxt, valid = simulate((2, 2, 1), "forward", grid)
        self.assertFalse(valid)
        self.assertEqual(nxt, (2, 2, 1))


if __name__ == "__main__":
    unittest.main()
