# Copyright (c) 2026 Blackridge Autonomy LLC. All rights reserved.
"""
Blackridge Swarm Governor SDK
==============================
WHACO (Warranted Hierarchical Autonomous Cooperative Operations)
Resource-aware distributed control governor with scale-emergent
attrition resilience and PX4 SITL validation.

Usage:
    from blackridge_governor import Governor

    gov = Governor(fuel_budget=4.0)
    thrust, mode_id, mode_name = gov.step(
        wind_x=-3.0, wind_y=0.5,
        fuel_remaining=2.8, dist_to_goal=45.0,
        groundspeed=6.2, throttle_pct=0.55,
    )
"""

from .adaptive_thresholds import AdaptiveGovernor, AdaptiveThresholdController
from .governor import MODES, THRUST_FRACS, Governor

__version__ = "1.2.1"
__all__ = [
    "Governor", "MODES", "THRUST_FRACS",
    "AdaptiveGovernor", "AdaptiveThresholdController",
]
