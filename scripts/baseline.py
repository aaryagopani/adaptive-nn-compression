"""
Baseline inference script — deterministic (no LLM) strategy.

Runs a fixed compression strategy against all 3 tasks to produce
reproducible baseline scores. Uses the same logging format as inference.py.

Strategy:
  1. Fuse all available Conv+BN groups (free size reduction)
  2. Quantize largest layers to 16-bit
  3. Quantize largest layers to 8-bit
  4. Light pruning (0.25 sparsity) on largest remaining layers
  5. Finish

Usage:
    python scripts/baseline.py
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "edge-ai-compression-gym"
MODEL_NAME = "deterministic-baseline"
TASKS = ["easy", "medium", "hard"]
REWARD_FN = "hybrid_a"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def env_reset(task_id: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "reward_fn": REWARD_FN},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def get_baseline_action(obs: Dict[str, Any], phase: List[str]) -> Dict[str, Any]:
    """Deterministic strategy: fuse -> quantize 16 -> quantize 8 -> prune -> finish."""

    # Phase 1: Fuse all available groups
    if "fuse" not in phase:
        for fg in obs.get("fusable_groups", []):
            if not fg.get("is_fused", False):
                return {"action_type": "fuse", "fuse_group_id": fg["group_id"]}
        phase.append("fuse")

    compressible = [
        l for l in obs.get("layers", [])
        if l.get("compressible", False) and not l.get("is_frozen", False)
    ]
    compressible.sort(key=lambda l: l["size_mb"], reverse=True)

    # Phase 2: Quantize to 16-bit
    if "q16" not in phase:
        for layer in compressible:
            if layer["current_bits"] > 16:
                return {
                    "action_type": "quantize",
                    "target_layer": layer["name"],
                    "target_bits": 16,
                }
        phase.append("q16")

    # Phase 3: Quantize to 8-bit
    if "q8" not in phase:
        for layer in compressible:
            if layer["current_bits"] > 8:
                return {
                    "action_type": "quantize",
                    "target_layer": layer["name"],
                    "target_bits": 8,
                }
        phase.append("q8")

    # Phase 4: Light pruning
    if "prune" not in phase:
        acc_budget = obs.get("accuracy_budget", 0.0)
        if acc_budget > 0.5:
            for layer in compressible:
                if layer["current_sparsity"] < 0.25:
                    return {
                        "action_type": "prune",
                        "target_layer": layer["name"],
                        "sparsity": 0.25,
                    }
        phase.append("prune")

    return {"action_type": "finish"}


def run_task(task_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    phase: List[str] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_data = env_reset(task_id)
        obs = reset_data.get("observation", reset_data)
        done = obs.get("done", False)
        max_steps = obs.get("max_steps", 20)

        for step in range(1, max_steps + 1):
            if done:
                break

            action = get_baseline_action(obs, phase)
            action_str = json.dumps(action, separators=(",", ":"))

            try:
                step_data = env_step(action)
                obs = step_data.get("observation", step_data)
                reward = step_data.get("reward", obs.get("reward", 0.0))
                done = step_data.get("done", obs.get("done", False))
                error = None
                last_result = obs.get("last_action_result", {})
                if not last_result.get("applied", True) and last_result.get("rejection_code"):
                    error = last_result["rejection_code"]
            except requests.HTTPError as e:
                reward = 0.0
                done = False
                error = str(e)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        if obs:
            comp_ratio = obs.get("compression_ratio", 1.0)
            target_comp = obs.get("target_compression", 2.0)
            acc_drop = obs.get("accuracy_drop", 0.0)
            max_acc_drop = obs.get("max_accuracy_drop", 5.0)
            acc_blown = acc_drop > max_acc_drop

            if acc_blown:
                score = 0.0
            else:
                compression_achievement = min(comp_ratio / target_comp, 1.0)
                accuracy_preservation = max(1.0 - (acc_drop / max_acc_drop), 0.0)
                score = compression_achievement * accuracy_preservation

        score = min(max(score, 0.0), 1.0)
        success = score >= 0.1

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    print(f"[INFO] Deterministic baseline agent", flush=True)
    print(f"[INFO] Environment: {ENV_URL}", flush=True)
    print(f"[INFO] Tasks: {TASKS}", flush=True)
    print("", flush=True)

    scores = {}
    for task_id in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_id}", flush=True)
        print(f"{'='*60}", flush=True)
        score = run_task(task_id)
        scores[task_id] = score
        print(f"\n[RESULT] task={task_id} score={score:.2f}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("Baseline Scores:", flush=True)
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.2f}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  average: {avg:.2f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
