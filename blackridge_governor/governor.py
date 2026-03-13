# Copyright (c) 2026 Blackridge Autonomy LLC. All rights reserved.
"""
WHACO Governor — Core mode-switching throttle controller.

WHACO (Warranted Hierarchical Autonomous Cooperative Operations)

Selects from 8 discrete thrust modes based on wind alignment,
fuel pressure, distance to goal, and look-ahead conditions.

Designed as a bolt-on module downstream of any route planner.

Compute budget: 15 microseconds per call.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Mode definitions: id -> name
MODES = {
    0: "RIDE",      # 20% — downwind glide
    1: "SPRINT",    # 85% — aggressive push
    2: "CRUISE",    # 55% — steady state
    3: "CONSERVE",  # 30% — fuel preservation
    4: "PUNCH",     # 100% — terminal/stall override
    5: "BOOST",     # 75% — moderate push
    6: "BRAKE",     # 40% — controlled deceleration
    7: "STABILIZE", # 55% — post-anomaly stabilize
}

# Mode -> thrust fraction [0.0, 1.0]
THRUST_FRACS = {
    "RIDE": 0.20, "SPRINT": 0.85, "CRUISE": 0.55, "CONSERVE": 0.30,
    "PUNCH": 1.00, "BOOST": 0.75, "BRAKE": 0.40, "STABILIZE": 0.55,
}

# Default thresholds (tunable via Governor constructor)
_DEFAULTS = {
    "tailwind_ride_threshold": 3.0,
    "headwind_sprint_threshold": 4.0,
    "fuel_conserve_threshold": 0.7,
    "fuel_critical_threshold": 0.9,
    "terminal_dist": 8.0,
    "terminal_headwind": 2.0,
    "stall_speed": 1.0,
    "stall_dist": 5.0,
    "stall_headwind": 2.0,
    "speed_floor": 2.0,
    "max_speed": 12.0,
}


@dataclass
class TelemetryFrame:
    """Single telemetry snapshot from the governor."""
    seq: int
    mode_id: int
    mode_name: str
    thrust_frac: float
    fuel_pressure: float
    fuel_remaining: float
    wind_alignment: float
    dist_to_goal: float
    groundspeed: float
    transition: bool = False
    prev_mode: int = -1


class Governor:
    """
    WHACO mode-switching throttle governor.

    Args:
        fuel_budget: Total fuel capacity (units). Default 4.0.
        thresholds: Optional dict overriding default mode selection thresholds.
    """

    def __init__(self, fuel_budget: float = 4.0,
                 thresholds: Optional[Dict[str, float]] = None):
        self.fuel_budget = fuel_budget
        self.thresholds = {**_DEFAULTS, **(thresholds or {})}

        # State
        self._mode_id = 2  # CRUISE
        self._prev_mode_id = -1
        self._seq = 0
        self._fuel_pressure = 0.0
        self._telemetry: List[TelemetryFrame] = []

    def step(self, wind_x: float = 0.0, wind_y: float = 0.0,
             fuel_remaining: float = 4.0, dist_to_goal: float = 50.0,
             groundspeed: float = 6.0, throttle_pct: float = 0.55,
             goal_dir_x: float = 1.0, goal_dir_y: float = 0.0,
             has_wind_cov: bool = True,
             wind_z: float = 0.0, goal_dir_z: float = 0.0,
             vertical_speed: float = 0.0) -> Tuple[float, int, str]:
        """
        Run one step of WHACO mode selection.

        Args:
            wind_x, wind_y: Wind vector (m/s). Negative x = headwind for +x goal.
            fuel_remaining: Current fuel level (same units as fuel_budget).
            dist_to_goal: Distance to goal (meters or units).
            groundspeed: Current ground speed (m/s).
            throttle_pct: Current throttle command [0, 1].
            goal_dir_x, goal_dir_y: Unit vector toward goal.
            has_wind_cov: True if wind is from EKF2, False to use proxy.
            wind_z: Vertical wind component (m/s). Positive = updraft.
            goal_dir_z: Vertical component of goal direction (for 3D).
            vertical_speed: Current vertical speed (m/s). Positive = ascending.

        Returns:
            (thrust_frac, mode_id, mode_name)
        """
        T = self.thresholds

        if dist_to_goal < 1e-6:
            return 0.55, 2, "CRUISE"

        # Normalize goal direction (2D or 3D)
        if abs(goal_dir_z) > 1e-6:
            gd = np.array([goal_dir_x, goal_dir_y, goal_dir_z])
        else:
            gd = np.array([goal_dir_x, goal_dir_y])
        gd_norm = np.linalg.norm(gd)
        if gd_norm > 1e-6:
            gd = gd / gd_norm

        # Wind alignment (3D-aware)
        if has_wind_cov:
            if len(gd) == 3:
                wind_alignment = (wind_x * gd[0] + wind_y * gd[1]
                                  + wind_z * gd[2])
            else:
                wind_alignment = wind_x * gd[0] + wind_y * gd[1]
        else:
            # Groundspeed-based proxy
            expected = T["max_speed"] * throttle_pct
            deficit = max(0, expected - groundspeed)
            wind_alignment = -deficit

        goal_headwind = max(0, -wind_alignment)

        # Fuel pressure model with gravity penalty for 3D
        speed = max(groundspeed, T["speed_floor"])
        steps_est = dist_to_goal / (speed * 0.5) if dist_to_goal > 0 else 1
        fps = 0.008 * (5.5)**2 * 0.5
        fuel_needed = fps * steps_est

        # Gravity penalty: ascending flight costs more fuel
        gravity_coeff = T.get("gravity_fuel_penalty", 0.002)
        net_climb = max(0.0, vertical_speed - wind_z)
        gravity_penalty = gravity_coeff * net_climb * steps_est
        fuel_needed += gravity_penalty

        fuel_ratio = fuel_remaining / max(fuel_needed, 0.001)
        self._fuel_pressure = max(0.0, min(1.0, 1.0 - fuel_ratio))

        # Stall detection override
        if (groundspeed < T["stall_speed"] and
                dist_to_goal > T["stall_dist"] and
                goal_headwind > T["stall_headwind"]):
            mode_id = 4  # PUNCH
        else:
            mode_id = self._select_mode(
                wind_alignment, fuel_ratio, dist_to_goal,
                goal_headwind,
            )

        mode_name = MODES[mode_id]
        thrust_frac = THRUST_FRACS[mode_name]

        # Telemetry
        transition = (mode_id != self._prev_mode_id)
        self._telemetry.append(TelemetryFrame(
            seq=self._seq, mode_id=mode_id, mode_name=mode_name,
            thrust_frac=thrust_frac, fuel_pressure=self._fuel_pressure,
            fuel_remaining=fuel_remaining, wind_alignment=wind_alignment,
            dist_to_goal=dist_to_goal, groundspeed=groundspeed,
            transition=transition, prev_mode=self._prev_mode_id,
        ))
        self._prev_mode_id = mode_id
        self._mode_id = mode_id
        self._seq += 1

        return thrust_frac, mode_id, mode_name

    def _select_mode(self, wind_alignment: float, fuel_ratio: float,
                     dist: float, goal_headwind: float) -> int:
        """Core mode selection logic."""
        T = self.thresholds

        # Terminal approach: PUNCH through headwind
        if dist < T["terminal_dist"] and goal_headwind > T["terminal_headwind"]:
            return 4  # PUNCH

        # Fuel critical: CONSERVE
        if self._fuel_pressure > T["fuel_critical_threshold"]:
            return 3  # CONSERVE

        # Strong tailwind: RIDE
        if wind_alignment > T["tailwind_ride_threshold"]:
            return 0  # RIDE

        # Strong headwind + fuel ok: SPRINT
        if goal_headwind > T["headwind_sprint_threshold"] and fuel_ratio > 1.5:
            return 1  # SPRINT

        # Moderate fuel pressure: CONSERVE
        if self._fuel_pressure > T["fuel_conserve_threshold"]:
            return 3  # CONSERVE

        # Default: CRUISE
        return 2  # CRUISE

    def reset(self):
        """Reset governor state for a new mission."""
        self._mode_id = 2
        self._prev_mode_id = -1
        self._seq = 0
        self._fuel_pressure = 0.0
        self._telemetry.clear()

    def telemetry(self) -> List[TelemetryFrame]:
        """Return captured telemetry frames."""
        return list(self._telemetry)

    @property
    def mode(self) -> str:
        """Current mode name."""
        return MODES[self._mode_id]

    @property
    def fuel_pressure(self) -> float:
        """Current fuel pressure [0, 1]."""
        return self._fuel_pressure
