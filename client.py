"""
Typed OpenEnv client for the Edge-AI Compression Gym.

Usage:
    async with CompressionEnvClient(base_url="http://localhost:7860") as client:
        result = await client.reset(task_id="easy")
        result = await client.step(
            CompressionAction(
                action_type="quantize",
                target_layer="features.0",
                target_bits=8,
            )
        )
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from openenv.core.env_client import EnvClient, StepResult

from models import CompressionAction, CompressionObservation, CompressionState


class CompressionEnvClient(
    EnvClient[CompressionAction, CompressionObservation, CompressionState]
):
    """Typed async client for the compression environment."""

    def _step_payload(self, action: CompressionAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self,
        payload: Dict[str, Any],
    ) -> StepResult[CompressionObservation]:
        observation_payload = payload.get("observation", payload)
        observation = CompressionObservation(**observation_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CompressionState:
        return CompressionState(**payload)


if __name__ == "__main__":
    import asyncio

    async def _smoke_test() -> None:
        async with CompressionEnvClient(base_url="http://localhost:7860") as client:
            result = await client.reset(task_id="easy")
            obs = result.observation
            print(f"Reset: {obs.model_name}")
            print(
                f"Baseline: {obs.baseline_mb:.2f}MB, {obs.baseline_accuracy:.2f}% "
                f"| target {obs.target_compression:.2f}x"
            )

            result = await client.step(
                CompressionAction(
                    action_type="quantize",
                    target_layer="features.0",
                    target_bits=16,
                )
            )
            obs = result.observation
            print(f"Reward: {obs.reward:.4f}")
            print(obs.last_action_result.debug_summary)

    asyncio.run(_smoke_test())
