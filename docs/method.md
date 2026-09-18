# PSH Planner: Predictive Safe Hybrid Navigation under Partial Observation

## Problem

The project studies goal-directed navigation on a partially observed grid with static geometry and moving obstacles. The robot has a discrete heading and can move forward, move backward, or rotate. At each control cycle it sees only nearby moving obstacles. A useful planner must therefore balance route efficiency, collision risk, and bounded online computation.

## Method

PSH decomposes the problem into four modules.

1. **Global corridor.** D* Lite maintains a route from the current position to the goal on the static map. The route is updated incrementally as the start advances.
2. **Motion prediction.** A nearest-neighbour tracker estimates one-cell velocities from consecutive local observations. Constant-velocity forecasts are converted into a short-horizon probabilistic occupancy map with increasing spatial uncertainty.
3. **Rolling-horizon control.** Beam search evaluates action sequences against goal progress, deviation from the global corridor, control effort, repeated visits, and predicted collision probability. Only the first action is executed before replanning.
4. **Safety shield.** A separate one-step filter rejects an action when its predicted occupancy risk crosses a threshold and selects the lowest-risk feasible fallback.

The local objective for a candidate sequence is

```text
J = sum_t [c_action + w_g d_goal + w_c d_corridor
           + w_r p_collision + w_v n_visit - w_p prior]
```

The optional `prior` term allows a learned policy, such as the existing DQN, to rank actions without granting it direct control. This preserves deterministic safety checks and makes the learned component replaceable.

## Research hypothesis

Under the same local observation and action budget, a forecast-aware local controller with an independent safety filter should reduce collision rate relative to reactive replanning. The ablation study tests prediction and shielding separately rather than treating the full stack as a black box.

## Experimental protocol

- 30 unseen seeds, numbered 20 through 49
- 24 by 32 maps with 34 static obstacle segments
- four moving obstacles with horizontal, vertical, and seeded random motion
- observation radius of five cells
- maximum 350 actions per episode
- identical reset state and seeded stochastic trajectory for every planner
- online A*, online D* Lite, two ablations, and full PSH

The benchmark reports success, collision, action count, traveled cells, turns, minimum obstacle clearance, mean planning latency, P95 latency, and shield interventions. Results are stored as episode-level CSV and aggregate JSON so every summary value can be recomputed.

## Scope and limitations

This is a grid-world research prototype. The motion model does not include acceleration, wheel slip, robot radius, communication delay, or continuous-time multi-agent interaction. The results therefore support the software architecture and its behavior in simulation; they do not establish performance on a physical robot. A real deployment would replace the tracker with state estimation, the discrete local search with a kinodynamic optimizer, and the one-cell collision model with footprint-aware continuous collision checking.
