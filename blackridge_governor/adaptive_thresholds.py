# Copyright (c) 2026 Blackridge Autonomy LLC. All rights reserved.
"""
Online Adaptive Thresholds — Gradient-free in-flight learning for WHACO Governor.

Uses a simplified diagonal CMA-ES to tune 5 governor thresholds in real-time
based on rolling fitness evaluation of swarm state (survival, fuel, connectivity,
mission progress).

Compute budget: <100μs per adaptation step.

Architecture:
    AdaptiveGovernor wraps Governor
        → every eval_window ticks: AdaptiveThresholdController.adapt()
            → DiagonalCMAES.ask() / tell()
                → updated thresholds → Governor

Safety constraints:
    - Hard bounds on all parameters
    - Rate-limited changes (max 10% per step)
    - Revert to defaults on 3 consecutive fitness drops >30%
    - Locked during CDO (no adaptation during crisis)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .governor import MODES, Governor

# ── Parameter Definitions ──────────────────────────────────────

@dataclass
class ThresholdParam:
    """Single tunable parameter with bounds and default."""
    name: str
    default: float
    low: float
    high: float


TUNABLE_PARAMS = [
    ThresholdParam("fuel_budget_threshold", 4.0, 2.0, 8.0),
    ThresholdParam("noise_sigma_threshold", 4.0, 2.0, 8.0),
    ThresholdParam("headwind_stall_trigger", 2.0, 1.0, 4.0),
    ThresholdParam("conserve_fuel_pressure", 0.5, 0.3, 0.8),
    ThresholdParam("punch_groundspeed_min", 1.0, 0.5, 2.0),
]

N_PARAMS = len(TUNABLE_PARAMS)


# ── Diagonal CMA-ES ───────────────────────────────────────────

class DiagonalCMAES:
    """
    Simplified CMA-ES with diagonal covariance for real-time use.

    Uses 1/5th success rule for step-size adaptation.
    No full covariance matrix — O(n) per step instead of O(n^2).

    Args:
        n_dim: Number of parameters.
        pop_size: Population size per generation.
        sigma0: Initial step size.
        bounds_low: Lower bounds for each parameter.
        bounds_high: Upper bounds for each parameter.
        rng: Numpy random generator.
    """

    def __init__(
        self,
        n_dim: int,
        pop_size: int = 5,
        sigma0: float = 0.3,
        bounds_low: Optional[np.ndarray] = None,
        bounds_high: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.n_dim = n_dim
        self.pop_size = pop_size
        self.rng = rng or np.random.default_rng(42)

        # Distribution parameters
        self.mean = np.zeros(n_dim)
        self.sigma = sigma0
        self.diag_cov = np.ones(n_dim)  # Diagonal covariance

        # Bounds
        self.bounds_low = bounds_low if bounds_low is not None else -np.inf * np.ones(n_dim)
        self.bounds_high = bounds_high if bounds_high is not None else np.inf * np.ones(n_dim)

        # 1/5th success rule state
        self._success_count = 0
        self._eval_count = 0
        self._generation = 0
        self._best_fitness = -np.inf

    def set_mean(self, mean: np.ndarray) -> None:
        """Set the current distribution mean (e.g., to current thresholds)."""
        self.mean = np.clip(mean, self.bounds_low, self.bounds_high)

    def ask(self) -> List[np.ndarray]:
        """
        Generate candidate solutions.

        Returns:
            List of candidate parameter vectors (pop_size candidates).
        """
        candidates = []
        for _ in range(self.pop_size):
            z = self.rng.standard_normal(self.n_dim)
            x = self.mean + self.sigma * np.sqrt(self.diag_cov) * z
            x = np.clip(x, self.bounds_low, self.bounds_high)
            candidates.append(x)
        return candidates

    def tell(self, candidates: List[np.ndarray], fitnesses: List[float]) -> None:
        """
        Update distribution based on evaluated fitness.

        Args:
            candidates: Parameter vectors from ask().
            fitnesses: Corresponding fitness values (higher = better).
        """
        if not candidates or not fitnesses:
            return

        # Sort by fitness (descending)
        pairs = sorted(zip(fitnesses, candidates), key=lambda x: x[0], reverse=True)

        # Weighted recombination (top half)
        mu = max(1, len(pairs) // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()

        # Update mean
        new_mean = np.zeros(self.n_dim)
        for i in range(mu):
            new_mean += weights[i] * pairs[i][1]
        new_mean = np.clip(new_mean, self.bounds_low, self.bounds_high)

        # Update diagonal covariance
        for i in range(mu):
            diff = pairs[i][1] - self.mean
            self.diag_cov = 0.8 * self.diag_cov + 0.2 * weights[i] * (diff ** 2) / max(self.sigma ** 2, 1e-10)
        self.diag_cov = np.clip(self.diag_cov, 0.01, 10.0)

        # 1/5th success rule for sigma
        best_gen_fitness = pairs[0][0]
        if best_gen_fitness > self._best_fitness:
            self._success_count += 1
            self._best_fitness = best_gen_fitness

        self._eval_count += 1
        if self._eval_count >= 5:
            success_rate = self._success_count / self._eval_count
            if success_rate > 0.2:
                self.sigma *= 1.1  # Explore more
            else:
                self.sigma *= 0.9  # Exploit more
            self.sigma = max(0.01, min(self.sigma, 1.0))
            self._success_count = 0
            self._eval_count = 0

        self.mean = new_mean
        self._generation += 1


# ── Adaptive Threshold Controller ──────────────────────────────

class AdaptiveThresholdController:
    """
    Manages in-flight adaptation of governor thresholds.

    Args:
        eval_window: Ticks between adaptation steps.
        fitness_weights: Dict of metric_name -> weight for fitness.
        max_rate: Maximum fractional change per adaptation step.
        revert_threshold: Fitness drop fraction triggering revert.
        revert_patience: Consecutive drops before reverting.
        rng: Numpy random generator.
    """

    def __init__(
        self,
        eval_window: int = 50,
        fitness_weights: Optional[Dict[str, float]] = None,
        max_rate: float = 0.10,
        revert_threshold: float = 0.30,
        revert_patience: int = 3,
        rng: Optional[np.random.Generator] = None,
    ):
        self.eval_window = eval_window
        self.max_rate = max_rate
        self.revert_threshold = revert_threshold
        self.revert_patience = revert_patience
        self.rng = rng or np.random.default_rng(42)

        self.fitness_weights = fitness_weights or {
            "survival_rate": 0.4,
            "fuel_efficiency": 0.2,
            "connectivity": 0.2,
            "mission_progress": 0.2,
        }

        # Initialize CMA-ES
        defaults = np.array([p.default for p in TUNABLE_PARAMS])
        bounds_low = np.array([p.low for p in TUNABLE_PARAMS])
        bounds_high = np.array([p.high for p in TUNABLE_PARAMS])

        self._cmaes = DiagonalCMAES(
            n_dim=N_PARAMS,
            pop_size=5,
            sigma0=0.2,
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            rng=self.rng,
        )
        self._cmaes.set_mean(defaults)

        # State
        self._defaults = defaults.copy()
        self._current = defaults.copy()
        self._previous = defaults.copy()
        self._tick = 0
        self._cdo_locked = False
        self._fitness_history: List[float] = []
        self._threshold_history: List[np.ndarray] = [defaults.copy()]
        self._baseline_fitness: Optional[float] = None
        self._consecutive_drops = 0

        # Candidate evaluation
        self._candidates: Optional[List[np.ndarray]] = None
        self._candidate_idx = 0
        self._candidate_fitnesses: List[float] = []

    @property
    def current_thresholds(self) -> Dict[str, float]:
        """Current threshold values as a dict."""
        return {p.name: float(self._current[i]) for i, p in enumerate(TUNABLE_PARAMS)}

    def compute_fitness(
        self,
        survival_rate: float,
        fuel_efficiency: float,
        connectivity: float,
        mission_progress: float,
    ) -> float:
        """
        Compute weighted fitness from swarm metrics.

        Args:
            survival_rate: Fraction of agents alive [0, 1].
            fuel_efficiency: Mean fuel remaining / initial [0, 1].
            connectivity: Algebraic connectivity (lambda2), clamped to [0, 1].
            mission_progress: Fraction of mission completed [0, 1].

        Returns:
            Scalar fitness value [0, 1].
        """
        w = self.fitness_weights
        fitness = (
            w["survival_rate"] * np.clip(survival_rate, 0, 1)
            + w["fuel_efficiency"] * np.clip(fuel_efficiency, 0, 1)
            + w["connectivity"] * np.clip(connectivity, 0, 1)
            + w["mission_progress"] * np.clip(mission_progress, 0, 1)
        )
        return float(fitness)

    def adapt(
        self,
        survival_rate: float,
        fuel_efficiency: float,
        connectivity: float,
        mission_progress: float,
    ) -> Dict[str, float]:
        """
        Run one adaptation step.

        Called every eval_window ticks. Evaluates current fitness,
        updates CMA-ES, generates new thresholds.

        Args:
            survival_rate: Fraction of agents alive.
            fuel_efficiency: Mean fuel remaining / initial.
            connectivity: Lambda2 of swarm graph.
            mission_progress: Mission completion fraction.

        Returns:
            Updated threshold dict.
        """
        self._tick += 1

        # CDO lock: no adaptation during crisis
        if self._cdo_locked:
            return self.current_thresholds

        fitness = self.compute_fitness(
            survival_rate, fuel_efficiency, connectivity, mission_progress
        )
        self._fitness_history.append(fitness)

        # Set baseline on first evaluation
        if self._baseline_fitness is None:
            self._baseline_fitness = fitness

        # Check for fitness degradation
        if self._baseline_fitness > 0:
            drop = (self._baseline_fitness - fitness) / self._baseline_fitness
            if drop > self.revert_threshold:
                self._consecutive_drops += 1
            else:
                self._consecutive_drops = 0

            # Revert if too many consecutive drops
            if self._consecutive_drops >= self.revert_patience:
                self._current = self._defaults.copy()
                self._cmaes.set_mean(self._defaults)
                self._consecutive_drops = 0
                self._threshold_history.append(self._current.copy())
                return self.current_thresholds

        # CMA-ES update cycle
        if self._candidates is None:
            # Start new generation
            self._candidates = self._cmaes.ask()
            self._candidate_idx = 0
            self._candidate_fitnesses = []

        # Evaluate current candidate
        self._candidate_fitnesses.append(fitness)
        self._candidate_idx += 1

        if self._candidate_idx >= len(self._candidates):
            # All candidates evaluated — update CMA-ES
            self._cmaes.tell(self._candidates, self._candidate_fitnesses)

            # Apply rate limiting
            new = self._cmaes.mean.copy()
            for i in range(N_PARAMS):
                max_delta = abs(self._current[i]) * self.max_rate
                delta = new[i] - self._current[i]
                if abs(delta) > max_delta:
                    new[i] = self._current[i] + np.sign(delta) * max_delta

            # Enforce bounds
            for i, p in enumerate(TUNABLE_PARAMS):
                new[i] = np.clip(new[i], p.low, p.high)

            self._previous = self._current.copy()
            self._current = new
            self._threshold_history.append(self._current.copy())

            # Reset for next generation
            self._candidates = None

        # Update baseline (moving average)
        self._baseline_fitness = 0.9 * self._baseline_fitness + 0.1 * fitness

        return self.current_thresholds

    def lock_for_cdo(self) -> None:
        """Lock adaptation during CDO — safety invariant."""
        self._cdo_locked = True

    def unlock_from_cdo(self) -> None:
        """Unlock adaptation when CDO exits."""
        self._cdo_locked = False

    def get_fitness_history(self) -> List[float]:
        """Return fitness evaluation history."""
        return list(self._fitness_history)

    def get_threshold_history(self) -> List[Dict[str, float]]:
        """Return threshold history as list of dicts."""
        return [
            {p.name: float(arr[i]) for i, p in enumerate(TUNABLE_PARAMS)}
            for arr in self._threshold_history
        ]

    def reset(self) -> None:
        """Reset all adaptation state."""
        self._current = self._defaults.copy()
        self._previous = self._defaults.copy()
        self._tick = 0
        self._cdo_locked = False
        self._fitness_history.clear()
        self._threshold_history = [self._defaults.copy()]
        self._baseline_fitness = None
        self._consecutive_drops = 0
        self._candidates = None
        self._cmaes.set_mean(self._defaults)


# ── Adaptive Governor ──────────────────────────────────────────

class AdaptiveGovernor:
    """
    Governor wrapper with online threshold adaptation.

    Runs the base Governor with dynamically tuned thresholds.
    Adaptation occurs every eval_window ticks when swarm
    metrics are provided.

    Args:
        fuel_budget: Initial fuel budget.
        eval_window: Ticks between adaptations.
        rng: Numpy random generator.
    """

    def __init__(
        self,
        fuel_budget: float = 4.0,
        eval_window: int = 50,
        rng: Optional[np.random.Generator] = None,
    ):
        self.governor = Governor(fuel_budget=fuel_budget)
        self.controller = AdaptiveThresholdController(
            eval_window=eval_window,
            rng=rng or np.random.default_rng(42),
        )
        self._eval_window = eval_window
        self._step_count = 0
        self._last_metrics: Optional[Dict[str, float]] = None

    def step(
        self,
        swarm_metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Tuple[float, int, str]:
        """
        Run one governor step with adaptive thresholds.

        Args:
            swarm_metrics: Optional dict with keys:
                survival_rate, fuel_efficiency, connectivity, mission_progress
            **kwargs: All args forwarded to Governor.step().

        Returns:
            (thrust_frac, mode_id, mode_name)
        """
        self._step_count += 1

        # Store latest metrics for adaptation
        if swarm_metrics is not None:
            self._last_metrics = swarm_metrics

        # Adapt thresholds periodically
        if (self._step_count % self._eval_window == 0
                and self._last_metrics is not None):
            new_thresholds = self.controller.adapt(
                survival_rate=self._last_metrics.get("survival_rate", 1.0),
                fuel_efficiency=self._last_metrics.get("fuel_efficiency", 1.0),
                connectivity=self._last_metrics.get("connectivity", 0.5),
                mission_progress=self._last_metrics.get("mission_progress", 0.0),
            )
            # Apply to governor
            self._apply_thresholds(new_thresholds)

        return self.governor.step(**kwargs)

    def _apply_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Map adaptive thresholds to governor threshold names."""
        mapping = {
            "fuel_budget_threshold": "fuel_conserve_threshold",
            "noise_sigma_threshold": "headwind_sprint_threshold",
            "headwind_stall_trigger": "stall_headwind",
            "conserve_fuel_pressure": "fuel_conserve_threshold",
            "punch_groundspeed_min": "stall_speed",
        }
        for adaptive_name, gov_name in mapping.items():
            if adaptive_name in thresholds:
                self.governor.thresholds[gov_name] = thresholds[adaptive_name]

    def lock_for_cdo(self) -> None:
        """Lock adaptation during CDO."""
        self.controller.lock_for_cdo()

    def unlock_from_cdo(self) -> None:
        """Unlock adaptation after CDO."""
        self.controller.unlock_from_cdo()

    def get_threshold_history(self) -> List[Dict[str, float]]:
        """Threshold evolution over time."""
        return self.controller.get_threshold_history()

    def get_fitness_history(self) -> List[float]:
        """Fitness evaluation history."""
        return self.controller.get_fitness_history()

    def reset(self) -> None:
        """Reset governor and controller."""
        self.governor.reset()
        self.controller.reset()
        self._step_count = 0
        self._last_metrics = None


# ── Standalone Tests ───────────────────────────────────────────

def _run_tests():
    """Adaptive thresholds validation — PASS/FAIL output."""
    passed = 0
    failed = 0
    total = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name}  {detail}")

    print("=" * 60)
    print("Adaptive Thresholds Tests")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # --- Test 1: CMA-ES basic operation ---
    cma = DiagonalCMAES(n_dim=5, pop_size=5, sigma0=0.3, rng=rng,
                        bounds_low=np.zeros(5), bounds_high=np.ones(5) * 10)
    cma.set_mean(np.array([4.0, 4.0, 2.0, 0.5, 1.0]))
    candidates = cma.ask()
    check("cmaes_pop_size", len(candidates) == 5)
    check("cmaes_dims", all(len(c) == 5 for c in candidates))
    check("cmaes_bounds", all(
        np.all(c >= 0) and np.all(c <= 10) for c in candidates
    ))

    # Provide fitness and update
    fitnesses = [rng.random() for _ in candidates]
    cma.tell(candidates, fitnesses)
    check("cmaes_generation", cma._generation == 1)

    # --- Test 2: Controller initialization ---
    ctrl = AdaptiveThresholdController(eval_window=10, rng=rng)
    thresholds = ctrl.current_thresholds
    check("ctrl_param_count", len(thresholds) == 5)
    check("ctrl_defaults", thresholds["fuel_budget_threshold"] == 4.0)

    # --- Test 3: Fitness computation ---
    fitness = ctrl.compute_fitness(1.0, 0.8, 0.5, 0.3)
    expected = 0.4 * 1.0 + 0.2 * 0.8 + 0.2 * 0.5 + 0.2 * 0.3
    check("fitness_value", abs(fitness - expected) < 1e-6,
          f"expected {expected:.4f}, got {fitness:.4f}")

    # --- Test 4: Adaptation changes thresholds ---
    ctrl2 = AdaptiveThresholdController(eval_window=1, rng=np.random.default_rng(123))
    initial = ctrl2.current_thresholds.copy()
    # Run several adaptation steps with varying metrics
    for i in range(20):
        ctrl2.adapt(
            survival_rate=0.9 - i * 0.01,
            fuel_efficiency=0.7,
            connectivity=0.4 + i * 0.01,
            mission_progress=i / 20.0,
        )
    final = ctrl2.current_thresholds
    changed = sum(1 for k in initial if abs(initial[k] - final[k]) > 1e-6)
    check("thresholds_adapted", changed > 0,
          f"only {changed}/5 parameters changed")
    check("threshold_history_length", len(ctrl2.get_threshold_history()) > 1)
    check("fitness_history_length", len(ctrl2.get_fitness_history()) == 20)

    # --- Test 5: Rate limiting ---
    ctrl3 = AdaptiveThresholdController(eval_window=1, max_rate=0.10,
                                         rng=np.random.default_rng(42))
    before = ctrl3._current.copy()
    ctrl3.adapt(1.0, 1.0, 1.0, 1.0)  # First eval sets baseline
    # Force a big jump by manipulating CMA-ES
    ctrl3._cmaes.mean = ctrl3._current * 2.0
    ctrl3._candidates = [ctrl3._current * 2.0] * 5
    ctrl3._candidate_idx = 0
    ctrl3._candidate_fitnesses = []
    for _ in range(5):
        ctrl3.adapt(1.0, 1.0, 1.0, 1.0)
    after = ctrl3._current
    max_change = max(abs(after[i] - before[i]) / max(abs(before[i]), 1e-6)
                     for i in range(N_PARAMS))
    check("rate_limited", max_change <= 0.15,  # Allow small tolerance
          f"max fractional change: {max_change:.4f}")

    # --- Test 6: CDO lock ---
    ctrl4 = AdaptiveThresholdController(eval_window=1, rng=rng)
    ctrl4.adapt(1.0, 0.5, 0.5, 0.5)  # Initial
    locked_thresholds = ctrl4.current_thresholds.copy()
    ctrl4.lock_for_cdo()
    ctrl4.adapt(0.2, 0.1, 0.1, 0.1)  # Should be ignored
    check("cdo_lock", ctrl4.current_thresholds == locked_thresholds)
    ctrl4.unlock_from_cdo()
    ctrl4.adapt(0.9, 0.9, 0.9, 0.9)
    check("cdo_unlock", True)  # No crash

    # --- Test 7: Revert on degradation ---
    ctrl5 = AdaptiveThresholdController(
        eval_window=1, revert_threshold=0.30, revert_patience=3,
        rng=np.random.default_rng(99),
    )
    # Good baseline
    ctrl5.adapt(1.0, 1.0, 1.0, 1.0)
    # Severe degradation
    for _ in range(5):
        ctrl5.adapt(0.1, 0.1, 0.1, 0.1)
    # Should have reverted to defaults
    defaults = {p.name: p.default for p in TUNABLE_PARAMS}
    reverted = ctrl5.current_thresholds
    match_count = sum(1 for k in defaults if abs(reverted[k] - defaults[k]) < 0.5)
    check("revert_on_degradation", match_count >= 3,
          f"{match_count}/5 near defaults")

    # --- Test 8: AdaptiveGovernor integration ---
    agov = AdaptiveGovernor(fuel_budget=4.0, eval_window=10, rng=rng)
    results = []
    for i in range(100):
        metrics = {
            "survival_rate": 0.9,
            "fuel_efficiency": 0.7,
            "connectivity": 0.5,
            "mission_progress": i / 100.0,
        }
        thrust, mode_id, mode_name = agov.step(
            swarm_metrics=metrics,
            wind_x=-2.0, wind_y=0.5,
            fuel_remaining=3.0, dist_to_goal=50.0 - i * 0.4,
            groundspeed=6.0, throttle_pct=0.55,
        )
        results.append((thrust, mode_id, mode_name))
    check("agov_runs_100_steps", len(results) == 100)
    check("agov_valid_thrust", all(0.0 <= r[0] <= 1.0 for r in results))
    check("agov_valid_modes", all(r[1] in MODES for r in results))

    # Thresholds should have evolved
    history = agov.get_threshold_history()
    check("agov_threshold_evolution", len(history) > 1,
          f"history length: {len(history)}")

    # --- Test 9: Bounds enforcement ---
    ctrl6 = AdaptiveThresholdController(eval_window=1, rng=rng)
    # Force extreme values
    ctrl6._current = np.array([1.0, 1.0, 0.5, 0.2, 0.3])
    for _ in range(50):
        ctrl6.adapt(0.5, 0.5, 0.5, 0.5)
    final = ctrl6._current
    for i, p in enumerate(TUNABLE_PARAMS):
        check(f"bounds_{p.name}",
              p.low <= final[i] <= p.high,
              f"{final[i]:.3f} not in [{p.low}, {p.high}]")

    print("=" * 60)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_tests() else 1)
