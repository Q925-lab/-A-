"""Short-horizon motion tracking and probabilistic occupancy prediction."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

Cell = Tuple[int, int]


@dataclass
class Track:
    position: Cell
    velocity: Cell = (0, 0)
    missed: int = 0


class DynamicObstacleTracker:
    """Track anonymous obstacles with nearest-neighbour data association.

    The model deliberately stays lightweight: it estimates a bounded grid
    velocity and spreads probability around the constant-velocity forecast.
    This is fast enough to run inside every planning cycle.
    """

    def __init__(self, horizon: int = 7, max_missed: int = 2):
        self.horizon = horizon
        self.max_missed = max_missed
        self.tracks: List[Track] = []

    def reset(self):
        self.tracks = []

    def update(self, observations: Iterable[Cell]):
        observations = list(observations)
        unmatched = set(range(len(observations)))

        for track in self.tracks:
            candidates = [
                (abs(track.position[0] - observations[i][0])
                 + abs(track.position[1] - observations[i][1]), i)
                for i in unmatched
            ]
            if candidates and min(candidates)[0] <= 2:
                _, idx = min(candidates)
                new_pos = observations[idx]
                dr = max(-1, min(1, new_pos[0] - track.position[0]))
                dc = max(-1, min(1, new_pos[1] - track.position[1]))
                track.velocity = (dr, dc)
                track.position = new_pos
                track.missed = 0
                unmatched.remove(idx)
            else:
                track.missed += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        self.tracks.extend(Track(observations[i]) for i in sorted(unmatched))

    def predict(self, shape: Tuple[int, int], static_grid) -> List[Dict[Cell, float]]:
        rows, cols = shape
        forecasts: List[Dict[Cell, float]] = [dict() for _ in range(self.horizon + 1)]
        for track in self.tracks:
            r, c = track.position
            vr, vc = track.velocity
            for t in range(self.horizon + 1):
                cr = r + vr * t
                cc = c + vc * t
                candidates = [((cr, cc), 0.68)]
                # Uncertainty grows with the forecast horizon.
                side_mass = min(0.09, 0.025 + 0.009 * t)
                candidates += [
                    ((cr - 1, cc), side_mass), ((cr + 1, cc), side_mass),
                    ((cr, cc - 1), side_mass), ((cr, cc + 1), side_mass),
                ]
                for (pr, pc), prob in candidates:
                    if 0 <= pr < rows and 0 <= pc < cols and static_grid[pr, pc] != 1:
                        forecasts[t][(pr, pc)] = min(
                            1.0, forecasts[t].get((pr, pc), 0.0) + prob
                        )
        return forecasts
