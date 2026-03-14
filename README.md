# WHACO — Warranted Hierarchical Autonomous Cooperative Operations

**Resource-aware distributed autonomy governor for energy-constrained multi-agent drone systems.**

Developed by [Blackridge Autonomy LLC](https://blackridgeautonomy.com).

[![PyPI](https://img.shields.io/pypi/v/blackridge-whaco)](https://pypi.org/project/blackridge-whaco/)
[![Python](https://img.shields.io/pypi/pyversions/blackridge-whaco)](https://pypi.org/project/blackridge-whaco/)
[![CI](https://github.com/morpheus18-glitch/blackridge-whaco/actions/workflows/ci.yml/badge.svg)](https://github.com/morpheus18-glitch/blackridge-whaco/actions)

## Overview

WHACO is a bolt-on throttle governor that sits downstream of any route planner and selects from **8 discrete thrust modes** based on wind alignment, fuel pressure, distance to goal, and look-ahead conditions.

### Core Capabilities

| Capability | Description |
|---|---|
| **8-Mode Throttle Governor** | State-dependent thrust selection for energy-constrained operations |
| **15 microsecond per-step budget** | Real-time capable on embedded hardware |
| **Online Adaptive Thresholds** | Diagonal CMA-ES tunes 5 governor thresholds in-flight |
| **3D Wind-Aware** | Full 3D wind alignment with gravity penalty modeling |
| **PX4 Compatible** | Designed for MAVLink companion integration at 50 Hz |

### Mode Table

| ID | Mode | Thrust | Purpose |
|----|------|--------|---------|
| 0 | RIDE | 20% | Downwind glide |
| 1 | SPRINT | 85% | Aggressive push against headwind |
| 2 | CRUISE | 55% | Steady-state default |
| 3 | CONSERVE | 30% | Fuel preservation |
| 4 | PUNCH | 100% | Terminal/stall override |
| 5 | BOOST | 75% | Moderate push |
| 6 | BRAKE | 40% | Controlled deceleration |
| 7 | STABILIZE | 55% | Post-anomaly stabilization |

## Installation

```bash
pip install blackridge-whaco
```

Optional dependency groups:

```bash
pip install blackridge-whaco[viz]   # + matplotlib
pip install blackridge-whaco[dev]   # + matplotlib, ruff, coverage
pip install blackridge-whaco[all]   # everything
```

## Quick Start

```python
from blackridge_governor import Governor

gov = Governor(fuel_budget=4.0)
thrust, mode_id, mode_name = gov.step(
    wind_x=-3.0, wind_y=0.5,
    fuel_remaining=2.8, dist_to_goal=45.0,
    groundspeed=6.2, throttle_pct=0.55,
)
print(f"Mode: {mode_name}, Thrust: {thrust:.0%}")
# Mode: SPRINT, Thrust: 85%
```

### Adaptive Governor (Online Threshold Tuning)

```python
from blackridge_governor import AdaptiveGovernor

agov = AdaptiveGovernor(fuel_budget=4.0, eval_window=50)

for step in range(1000):
    metrics = {
        "survival_rate": 0.95,
        "fuel_efficiency": 0.7,
        "connectivity": 0.5,
        "mission_progress": step / 1000.0,
    }
    thrust, mode_id, mode_name = agov.step(
        swarm_metrics=metrics,
        wind_x=-2.0, fuel_remaining=3.0,
        dist_to_goal=50.0, groundspeed=6.0,
    )
```

## Architecture

```
┌──────────────────────────────────┐
│  Governor (per-agent, 15us)      │
│  8-mode throttle selection       │
│  Fuel pressure model             │
│  Stall detection override        │
│  3D wind alignment               │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  AdaptiveThresholdController     │
│  Diagonal CMA-ES, eval_window    │
│  Rate-limited, CDO-locked        │
│  Safety revert on degradation    │
└──────────────────────────────────┘
```

## Testing

```bash
# SDK smoke test
python -m blackridge_governor.demo --quick

# Adaptive thresholds tests
python -m blackridge_governor.adaptive_thresholds
```

## API Reference

### `Governor(fuel_budget=4.0, thresholds=None)`

Core throttle governor. Call `.step()` each tick to get `(thrust_frac, mode_id, mode_name)`.

**Key parameters for `.step()`:**
- `wind_x`, `wind_y`, `wind_z` — Wind vector (m/s)
- `fuel_remaining` — Current fuel level
- `dist_to_goal` — Distance to goal (meters)
- `groundspeed` — Current ground speed (m/s)
- `has_wind_cov` — `True` for EKF2 wind, `False` for groundspeed proxy

### `AdaptiveGovernor(fuel_budget=4.0, eval_window=50)`

Governor wrapper with online CMA-ES threshold adaptation. Pass `swarm_metrics` dict to `.step()` for in-flight tuning.

### `AdaptiveThresholdController(eval_window=50, ...)`

Standalone threshold controller using diagonal CMA-ES with safety constraints (rate limiting, CDO lock, degradation revert).

## License

Proprietary. Copyright (c) 2026 Blackridge Autonomy LLC. All rights reserved.

See [LICENSE](LICENSE) for details. For licensing inquiries: austin@blackridgeautonomy.com
