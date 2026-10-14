"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") #groq api key"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "llama-3.3-70b-versatile"

ENV_URL = os.getenv("ENV_URL", "https://YOUR-ENVIRONMENT.hf.space/")
BENCHMARK = "edge-ai-compression-gym"

TASKS = ["easy", "medium", "hard"]
REWARD_FN = "hybrid_a"
TEMPERATURE = 0.4
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.1


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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def env_reset(task_id: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "reward_fn": REWARD_FN},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action": action},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def format_observation_for_llm(obs: Dict[str, Any]) -> str:
    layers_summary = []
    for layer in obs.get("layers", []):
        if layer.get("compressible", False):
            layers_summary.append(
                f"  - {layer['name']} ({layer['layer_type']}): "
                f"{layer['param_count']} params, {layer['size_mb']:.4f}MB, "
                f"bits={layer['current_bits']}, sparsity={layer['current_sparsity']:.2f}"
            )

    fuse_summary = []
    for fg in obs.get("fusable_groups", []):
        if not fg.get("is_fused", False):
            fuse_summary.append(
                f"  - {fg['group_id']}: {fg['pattern']} "
                f"(saves ~{fg['estimated_saving_mb']:.4f}MB)"
            )

    last_result = obs.get("last_action_result", {})
    feedback = ""
    if last_result.get("applied"):
        feedback = (
            f"Last action: APPLIED, dsize={last_result.get('delta_size_mb', 0):.4f}MB, "
            f"dacc={last_result.get('delta_accuracy', 0):.2f}%"
        )
    elif last_result.get("rejection_code"):
        feedback = (
            f"Last action: REJECTED ({last_result['rejection_code']}): "
            f"{last_result.get('rejection_detail', '')}"
        )

    return textwrap.dedent(
        f"""\
        Model: {obs['model_name']}
        Step: {obs['step']}/{obs['max_steps']} (remaining: {obs['steps_remaining']})
        Current size: {obs['current_mb']:.4f}MB (baseline: {obs['baseline_mb']:.4f}MB)
        Current accuracy: {obs['current_accuracy']:.2f}% (baseline: {obs['baseline_accuracy']:.2f}%)
        Compression ratio: {obs['compression_ratio']:.3f}x (target: {obs['target_compression']:.1f}x)
        Accuracy drop: {obs['accuracy_drop']:.2f}% (budget: {obs['accuracy_budget']:.2f}%, max: {obs['max_accuracy_drop']:.1f}%)
        Size budget remaining: {obs['size_budget_mb']:.4f}MB
        Valid actions: {obs.get('valid_action_types', [])}
        {feedback}

        Compressible layers:
        {chr(10).join(layers_summary) if layers_summary else "  (none)"}

        Unfused groups:
        {chr(10).join(fuse_summary) if fuse_summary else "  (none)"}
        """
    )


SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an AI agent compressing a PyTorch neural network for edge deployment.
    You must shrink the model to meet a compression target while preserving accuracy.

    Available actions (respond with EXACTLY one JSON object, no other text):

    1. Quantize a layer:
       {"action_type": "quantize", "target_layer": "<layer_name>", "target_bits": <4|8|16>}

    2. Prune a layer:
       {"action_type": "prune", "target_layer": "<layer_name>", "sparsity": <0.05-0.95>}

    3. Fuse layers:
       {"action_type": "fuse", "fuse_group_id": "<group_id>"}

    4. Finish episode:
       {"action_type": "finish"}

    Strategy rules:
    - Your primary goal is to maximize final score, not to keep acting.
    - Never keep pruning the same layer repeatedly once reward becomes very small or the accuracy budget is tightening.
    - Prefer quantization over aggressive pruning on medium and hard tasks.
    - Use small or moderate pruning only when the remaining accuracy budget is comfortably positive.
    - If accuracy_budget is low, avoid further pruning and consider finishing.
    - If compression_ratio is already close to target and additional actions are risky, finish.
    - Repeatedly increasing sparsity on the same layer is dangerous and often causes failure.
    - If the previous action had poor reward or pushed the budget down sharply, choose a safer action or finish.
    - When uncertain, choose finish instead of a risky prune.

    Output rules:
    - Return exactly one valid JSON object.
    - Do not include markdown, explanation, backticks, comments, or extra text.
    - Ensure the JSON is syntactically valid.

    Respond with only the JSON object.
    """
).strip()


def get_llm_action(client: OpenAI, obs_text: str, history: List[str]) -> Dict[str, Any]:
    history_block = "\n".join(history[-6:]) if history else "No prior actions."
    user_prompt = f"""{obs_text}

Action history (recent):
{history_block}

Respond with a single JSON action object:"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        if raw.startswith("```"):
            lines = raw.split("\n")
            json_lines = []
            inside = False
            for line in lines:
                if line.startswith("```") and not inside:
                    inside = True
                    continue
                if line.startswith("```") and inside:
                    break
                if inside:
                    json_lines.append(line)
            raw = "\n".join(json_lines)

        return json.loads(raw)
    except Exception as exc:
        print(f"Model request failed: {exc}", file=sys.stderr, flush=True)
        return {"action_type": "finish"}


def run_task(client: OpenAI, task_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_data = env_reset(task_id)
        obs = reset_data.get("observation", reset_data)
        done = obs.get("done", False)
        max_steps = obs.get("max_steps", 20)

        for step in range(1, max_steps + 1):
            if done:
                break

            obs_text = format_observation_for_llm(obs)
            action = get_llm_action(client, obs_text, history)
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
            except requests.HTTPError as exc:
                reward = 0.0
                done = False
                error = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            action_type = action.get("action_type", "unknown")
            result_info = f"reward={reward:.2f}"
            if error:
                result_info += f", error={error}"
            history.append(f"Step {step}: {action_type} -> {result_info}")

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
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        print(f"Task {task_id} failed: {exc}", file=sys.stderr, flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    if not API_KEY:
        print("HF_TOKEN environment variable is required.", file=sys.stderr, flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in TASKS:
        run_task(client, task_id)


if __name__ == "__main__":
    main()
