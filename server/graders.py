"""
Final reward logic for the Edge-AI Compression Gym.

The dev branch keeps only the selected reward function, `hybrid_a`, which
performed best in the local stress sweep.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


REJECTION_PENALTY = -0.10
RECOMPRESS_DECAY = 0.5
ACCURACY_BLOW_PENALTY = -1.0
NOOP_FINISH_PENALTY = -0.20


def _diminishing_factor(compression_history: List[str]) -> float:
    """Decay reward for repeated operations on the same layer."""
    return RECOMPRESS_DECAY ** len(compression_history)


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) < 1e-12:
        return default
    return num / den


class RewardFunction(ABC):
    """Base class for the environment reward."""

    name: str = "base"

    @abstractmethod
    def compute_step_reward(
        self,
        delta_size_mb: float,
        delta_accuracy: float,
        baseline_mb: float,
        current_mb: float,
        target_mb: float,
        accuracy_drop: float,
        max_accuracy_drop: float,
        compression_ratio: float,
        target_compression: float,
        steps_remaining: int,
        max_steps: int,
        compression_history: List[str],
        action_applied: bool,
        action_type: str,
        layer_info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute per-step reward in [-1, 1]."""

    @abstractmethod
    def compute_finish_bonus(
        self,
        compression_ratio: float,
        target_compression: float,
        accuracy_drop: float,
        max_accuracy_drop: float,
        steps_remaining: int,
        max_steps: int,
        target_met: bool,
    ) -> float:
        """Compute end-of-episode bonus in [-1, 1]."""


class HybridA(RewardFunction):
    """
    Final selected reward.

    Reward = budget-relative size gain x sigmoid accuracy gate x diminishing
    returns for repeated compression on the same layer.
    """

    name = "hybrid_a"

    def compute_step_reward(
        self,
        delta_size_mb: float,
        delta_accuracy: float,
        baseline_mb: float,
        current_mb: float,
        target_mb: float,
        accuracy_drop: float,
        max_accuracy_drop: float,
        compression_ratio: float,
        target_compression: float,
        steps_remaining: int,
        max_steps: int,
        compression_history: List[str],
        action_applied: bool,
        action_type: str,
        layer_info: Optional[Dict[str, Any]] = None,
    ) -> float:
        if not action_applied:
            return REJECTION_PENALTY

        size_budget_before = (current_mb - delta_size_mb) - target_mb
        size_gain = _clamp(
            _safe_div(abs(delta_size_mb), max(size_budget_before, 0.001)),
            0.0,
            1.0,
        )

        budget_used = _safe_div(accuracy_drop, max_accuracy_drop)
        accuracy_gate = 1.0 / (1.0 + math.exp(10.0 * (budget_used - 0.8)))

        diminish = _diminishing_factor(compression_history)
        reward = size_gain * accuracy_gate * diminish
        return _clamp(reward)

    def compute_finish_bonus(
        self,
        compression_ratio: float,
        target_compression: float,
        accuracy_drop: float,
        max_accuracy_drop: float,
        steps_remaining: int,
        max_steps: int,
        target_met: bool,
    ) -> float:
        if target_met:
            acc_preserved = 1.0 - _safe_div(accuracy_drop, max_accuracy_drop)
            over_compress = min(_safe_div(compression_ratio, target_compression), 1.5)
            return _clamp(acc_preserved * over_compress)

        gap = 1.0 - _safe_div(compression_ratio, target_compression)
        return _clamp(NOOP_FINISH_PENALTY * max(gap, 0.0))


REWARD_FUNCTIONS: Dict[str, type[RewardFunction]] = {
    "hybrid_a": HybridA,
}


def get_reward_function(name: str) -> RewardFunction:
    """Instantiate the final selected reward function."""
    if name not in REWARD_FUNCTIONS:
        raise ValueError("Only 'hybrid_a' is supported on the dev branch.")
    return HybridA()


def compute_final_score(
    compression_ratio: float,
    target_compression: float,
    accuracy_drop: float,
    max_accuracy_drop: float,
    accuracy_blown: bool,
) -> float:
    """Final episode score for the grader in [0.0, 1.0]."""
    if accuracy_blown:
        return 0.0

    compression_achievement = min(_safe_div(compression_ratio, target_compression), 1.0)
    accuracy_preservation = max(1.0 - _safe_div(accuracy_drop, max_accuracy_drop), 0.0)
    return compression_achievement * accuracy_preservation
