"""
environment.py — Core game loop for the Edge-AI Compression Gym.

Implements the OpenEnv Environment interface:
  - reset(task_id) → loads model, computes baseline, returns initial observation
  - step(action)   → applies compression op, evaluates, computes reward
  - state property  → returns CompressionState

Compression operations:
  - Quantize: Simulated bit-width reduction (scales size, adds calibrated accuracy noise)
  - Prune:    Real torch.nn.utils.prune.l1_unstructured + accuracy re-eval
  - Fuse:     Simulated Conv+BN(+ReLU) fusion (removes BN params from size)
  - Finish:   Ends episode early, triggers final scoring
"""

from __future__ import annotations

import copy
import math
import os
import random
import shutil
import tarfile
import uuid
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms

from openenv.core.env_server import Environment

# ── Adjust import paths so we can find sibling modules ──────────────────────
import sys
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SERVER_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from models import (
    CompressionAction, CompressionObservation, CompressionState,
    LayerInfo, FuseGroup, ActionResult, ActionRecord,
    LayerStatus, FuseGroupStatus,
)
from server.model_architectures import load_model
from server.graders import (
    get_reward_function,
    RewardFunction,
    ACCURACY_BLOW_PENALTY,
    NOOP_FINISH_PENALTY,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Task Configurations
# ═══════════════════════════════════════════════════════════════════════════════

TASK_CONFIGS: Dict[str, dict] = {
    "easy": {
        "model_key": "cnn",
        "target_compression": 1.5,
        "max_accuracy_drop": 5.0,   # percentage points
        "max_steps": 15,
    },
    "medium": {
        "model_key": "resnet",
        "target_compression": 2.0,
        "max_accuracy_drop": 3.0,
        "max_steps": 20,
    },
    "hard": {
        "model_key": "transformer",
        "target_compression": 2.0,
        "max_accuracy_drop": 2.0,
        "max_steps": 20,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

NON_COMPRESSIBLE_TYPES = frozenset({
    "BatchNorm1d", "BatchNorm2d", "Dropout", "Dropout2d",
    "LayerNorm", "MaxPool2d", "AdaptiveAvgPool2d",
    "ReLU", "GELU", "Identity",
})
MIN_COMPRESSIBLE_PARAMS = 100
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RUNTIME_DATA_DIR = os.path.join(PROJECT_DIR, ".runtime_data", "cifar_cache")
CIFAR_ARCHIVE_NAME = "cifar-10-python.tar.gz"
CIFAR_EXTRACTED_DIR = "cifar-10-batches-py"
EVAL_SAMPLES = 1000          # fast eval (~2s on CPU)
EVAL_BATCH_SIZE = 100

# CIFAR-10 normalization (must match training transforms)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _is_compressible(module: nn.Module) -> bool:
    """Check if a leaf module is eligible for quantize/prune."""
    type_name = type(module).__name__
    if type_name in NON_COMPRESSIBLE_TYPES:
        return False
    # Skip container modules (they have children)
    if len(list(module.children())) > 0:
        return False
    param_count = sum(p.numel() for p in module.parameters())
    return param_count >= MIN_COMPRESSIBLE_PARAMS


def _layer_param_count(module: nn.Module) -> int:
    """Count trainable parameters directly on this module."""
    return sum(p.numel() for p in module.parameters())


def _layer_size_mb(param_count: int, bits: int = 32, sparsity: float = 0.0) -> float:
    """Effective size in MB accounting for bit-width and sparsity."""
    return param_count * (bits / 8) * (1.0 - sparsity) / (1024 * 1024)


def _total_model_params(model: nn.Module) -> int:
    """Total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def _compute_model_size_mb(layer_statuses: List[LayerStatus]) -> float:
    """Sum effective sizes across all tracked layers."""
    total = 0.0
    for ls in layer_statuses:
        total += _layer_size_mb(ls.param_count, ls.current_bits, ls.current_sparsity)
    return total


def _quantization_noise(
    layer_name: str, old_bits: int, new_bits: int,
    param_count: int, total_params: int,
) -> float:
    """Deterministic accuracy penalty (in %-points) for simulated quantization."""
    seed = hash(f"{layer_name}_{old_bits}_{new_bits}") & 0xFFFFFFFF
    rng = random.Random(seed)
    base_factor = rng.uniform(0.3, 1.0)

    # Nonlinear severity — INT4 hurts much more than INT8
    severity_map = {16: 0.2, 8: 0.6, 4: 1.4}
    severity = severity_map.get(new_bits, 0.5)

    importance = param_count / max(total_params, 1)
    # Calibrated: quantizing ALL layers to INT8 ≈ 1.5-3% drop
    return base_factor * severity * importance * 5.0


def _find_fusable_groups(model: nn.Module) -> List[dict]:
    """Scan model for Conv2d → BatchNorm2d (→ ReLU) patterns."""
    groups: List[dict] = []
    group_id = 0

    for parent_name, parent_module in model.named_modules():
        children_list = list(parent_module.named_children())
        if len(children_list) < 2:
            continue

        used: set = set()
        for i in range(len(children_list)):
            if i in used:
                continue
            c_name, c_mod = children_list[i]
            if not isinstance(c_mod, nn.Conv2d):
                continue

            prefix = f"{parent_name}.{c_name}" if parent_name else c_name

            # Check next sibling for BatchNorm2d
            if i + 1 < len(children_list):
                n_name, n_mod = children_list[i + 1]
                if isinstance(n_mod, nn.BatchNorm2d):
                    bn_prefix = f"{parent_name}.{n_name}" if parent_name else n_name

                    # Check for ReLU after BN
                    if (i + 2 < len(children_list)
                            and isinstance(children_list[i + 2][1], nn.ReLU)):
                        r_name = children_list[i + 2][0]
                        r_prefix = f"{parent_name}.{r_name}" if parent_name else r_name
                        groups.append({
                            "group_id": f"fuse_{group_id}",
                            "layer_names": [prefix, bn_prefix, r_prefix],
                            "pattern": "Conv2d → BatchNorm2d → ReLU",
                        })
                        used.update({i, i + 1, i + 2})
                        group_id += 1
                    else:
                        groups.append({
                            "group_id": f"fuse_{group_id}",
                            "layer_names": [prefix, bn_prefix],
                            "pattern": "Conv2d → BatchNorm2d",
                        })
                        used.update({i, i + 1})
                        group_id += 1

    return groups


def _prepare_runtime_cifar_root() -> str:
    """
    Prepare a writable CIFAR-10 root for local evaluation.

    The checked-in ``data/`` directory may contain a partial or read-only
    extraction, so we extract into a runtime-owned cache directory instead.
    """
    os.makedirs(RUNTIME_DATA_DIR, exist_ok=True)

    return RUNTIME_DATA_DIR


def _cifar_archive_candidates() -> List[str]:
    return [
        os.path.join(RUNTIME_DATA_DIR, CIFAR_ARCHIVE_NAME),
        os.path.join(DATA_DIR, CIFAR_ARCHIVE_NAME),
    ]


def _has_usable_cifar_extraction(root: str) -> bool:
    test_batch = os.path.join(root, CIFAR_EXTRACTED_DIR, "test_batch")
    return os.path.isfile(test_batch)


def _extract_cifar_archive(archive_path: str, root: str) -> None:
    """
    Extract the CIFAR archive using plain file writes.

    ``torchvision`` uses ``tar.extractall()``, which can preserve archive modes
    that become unreadable on Windows. This extractor recreates files with the
    current user's default permissions instead.
    """
    os.makedirs(root, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_name = member.name.replace("\\", "/")
            target_path = os.path.abspath(os.path.join(root, member_name))
            root_abs = os.path.abspath(root)
            if not target_path.startswith(root_abs):
                raise RuntimeError(f"Unsafe archive path detected: {member.name}")

            if member.isdir():
                os.makedirs(target_path, exist_ok=True)
                continue

            extracted_file = tar.extractfile(member)
            if extracted_file is None:
                continue

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with extracted_file, open(target_path, "wb") as output_file:
                shutil.copyfileobj(extracted_file, output_file)


# ═══════════════════════════════════════════════════════════════════════════════
# CIFAR-10 Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

_cached_val_dataset = None


def _get_val_dataset():
    """Load (and cache) the CIFAR-10 test set."""
    global _cached_val_dataset
    if _cached_val_dataset is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        runtime_root = _prepare_runtime_cifar_root()
        last_error: Optional[Exception] = None

        for root in (runtime_root, DATA_DIR):
            if not _has_usable_cifar_extraction(root):
                continue
            try:
                _cached_val_dataset = torchvision.datasets.CIFAR10(
                    root=root,
                    train=False,
                    download=False,
                    transform=transform,
                )
                break
            except (PermissionError, OSError, RuntimeError) as exc:
                last_error = exc
        else:
            extracted = False
            for archive_path in _cifar_archive_candidates():
                if not os.path.exists(archive_path):
                    continue
                try:
                    _extract_cifar_archive(archive_path, runtime_root)
                    _cached_val_dataset = torchvision.datasets.CIFAR10(
                        root=runtime_root,
                        train=False,
                        download=False,
                        transform=transform,
                    )
                    extracted = True
                    break
                except (PermissionError, OSError, RuntimeError, tarfile.TarError) as exc:
                    last_error = exc

            if not extracted:
                try:
                    _cached_val_dataset = torchvision.datasets.CIFAR10(
                        root=runtime_root,
                        train=False,
                        download=True,
                        transform=transform,
                    )
                except (PermissionError, OSError, RuntimeError) as exc:
                    last_error = exc
                    raise RuntimeError(
                        "Unable to initialize the CIFAR-10 validation set from the "
                        "available local roots."
                    ) from last_error

    return _cached_val_dataset


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, num_samples: int = EVAL_SAMPLES) -> float:
    """Quick accuracy eval on CIFAR-10 validation subset. Returns percentage."""
    model.eval()
    dataset = _get_val_dataset()

    # Deterministic subset for reproducibility
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=g)[:num_samples].tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
    )

    correct = 0
    total = 0
    for images, labels in loader:
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# CompressionEnvironment
# ═══════════════════════════════════════════════════════════════════════════════

class CompressionEnvironment(Environment):
    """
    OpenEnv Environment for the Edge-AI Compression Gym.

    The agent compresses a pre-trained CIFAR-10 model using quantize, prune,
    fuse, and finish actions. Rewards come from a pluggable reward function.

    NOTE: OpenEnv creates a new instance per request, so we use class-level
    variables to persist state across reset/step calls.
    """

    _shared_state: Optional[CompressionState] = None
    _shared_model: Optional[nn.Module] = None
    _shared_reward_fn: Optional["RewardFunction"] = None
    _shared_modules_map: Dict[str, nn.Module] = {}
    _shared_total_params: int = 0

    def __init__(self):
        super().__init__()
        self._state = CompressionEnvironment._shared_state
        self._model = CompressionEnvironment._shared_model
        self._reward_fn = CompressionEnvironment._shared_reward_fn
        self._modules_map = CompressionEnvironment._shared_modules_map
        self._total_params = CompressionEnvironment._shared_total_params

    def _sync_to_class(self):
        """Persist instance state to class-level so the next instance picks it up."""
        CompressionEnvironment._shared_state = self._state
        CompressionEnvironment._shared_model = self._model
        CompressionEnvironment._shared_reward_fn = self._reward_fn
        CompressionEnvironment._shared_modules_map = self._modules_map
        CompressionEnvironment._shared_total_params = self._total_params

    # ────────────────────────────────────────────────────────────────────
    # reset
    # ────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> CompressionObservation:
        task_id = kwargs.get("task_id", "easy")
        requested_reward_fn = kwargs.get("reward_fn")

        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id '{task_id}'. Options: {list(TASK_CONFIGS.keys())}")

        if requested_reward_fn not in (None, "hybrid_a"):
            raise ValueError("The dev branch supports only the 'hybrid_a' reward function.")

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        cfg = TASK_CONFIGS[task_id]
        self._reward_fn = get_reward_function("hybrid_a")

        # Load fresh model
        self._model, baseline_accuracy, model_name = load_model(cfg["model_key"])
        self._model.eval()
        self._total_params = _total_model_params(self._model)
        self._modules_map = dict(self._model.named_modules())

        # Build layer status from model inspection
        layer_statuses = self._build_layer_statuses()
        baseline_mb = _compute_model_size_mb(layer_statuses)
        target_mb = baseline_mb / cfg["target_compression"]

        # Detect fusable groups
        raw_groups = _find_fusable_groups(self._model)
        fuse_group_statuses = []
        for g in raw_groups:
            saving = self._estimate_fuse_saving(g["layer_names"], layer_statuses)
            fuse_group_statuses.append(FuseGroupStatus(
                group_id=g["group_id"],
                layer_names=g["layer_names"],
                pattern=g["pattern"],
                is_fused=False,
                estimated_saving_mb=saving,
            ))

        # Verify baseline accuracy with a quick eval
        measured_accuracy = evaluate_accuracy(self._model)

        self._state = CompressionState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            model_name=model_name,
            model_key=cfg["model_key"],
            max_steps=cfg["max_steps"],
            target_compression=cfg["target_compression"],
            max_accuracy_drop=cfg["max_accuracy_drop"],
            baseline_mb=baseline_mb,
            baseline_accuracy=measured_accuracy,
            current_mb=baseline_mb,
            current_accuracy=measured_accuracy,
            cumulative_reward=0.0,
            is_done=False,
            game_over_reason=None,
            layer_status=layer_statuses,
            fuse_groups=fuse_group_statuses,
            action_history=[],
        )

        self._sync_to_class()

        return self._build_observation(
            step_reward=0.0,
            action_result=ActionResult(applied=False),
        )

    # ────────────────────────────────────────────────────────────────────
    # step
    # ────────────────────────────────────────────────────────────────────

    def step(
        self,
        action: CompressionAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> CompressionObservation:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._state.is_done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        s = self._state
        s.step_count += 1

        # Snapshot before
        size_before = s.current_mb
        acc_before = s.current_accuracy
        reward_context = self._snapshot_reward_context(action)

        # Dispatch action
        if action.action_type == "quantize":
            result = self._apply_quantize(action)
        elif action.action_type == "prune":
            result = self._apply_prune(action)
        elif action.action_type == "fuse":
            result = self._apply_fuse(action)
        elif action.action_type == "finish":
            result = self._apply_finish(action)
        else:
            result = ActionResult(
                applied=False,
                rejection_code="INVALID_ACTION_TYPE",
                rejection_detail=f"Unknown action type: {action.action_type}",
            )

        # Recompute model size
        s.current_mb = _compute_model_size_mb(s.layer_status)
        delta_size = s.current_mb - size_before
        delta_acc = s.current_accuracy - acc_before

        # Check accuracy blow-up
        accuracy_drop = s.baseline_accuracy - s.current_accuracy
        if accuracy_drop > s.max_accuracy_drop and not s.is_done:
            s.is_done = True
            s.game_over_reason = "accuracy_blown"
            step_reward = ACCURACY_BLOW_PENALTY
        elif s.is_done:
            # finish or steps exhausted already handled
            step_reward = self._compute_finish_reward(result)
        else:
            # Compute step reward
            step_reward = self._compute_step_reward(
                action, result, delta_size, delta_acc, reward_context,
            )

        # Check steps exhausted
        steps_remaining = s.max_steps - s.step_count
        if steps_remaining <= 0 and not s.is_done:
            s.is_done = True
            s.game_over_reason = "steps_exhausted"
            # Add finish bonus for steps exhausted
            bonus = self._compute_auto_finish_bonus()
            step_reward += bonus

        s.cumulative_reward += step_reward

        # Record action
        s.action_history.append(ActionRecord(
            step=s.step_count,
            action_type=action.action_type,
            target_layer=action.target_layer,
            target_bits=action.target_bits,
            sparsity=action.sparsity,
            fuse_group_id=action.fuse_group_id,
            result=result,
            reward=step_reward,
        ))

        self._sync_to_class()

        return self._build_observation(step_reward, result)

    # ────────────────────────────────────────────────────────────────────
    # state property
    # ────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> CompressionState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    # ═══════════════════════════════════════════════════════════════════
    # Action handlers
    # ═══════════════════════════════════════════════════════════════════

    def _apply_quantize(self, action: CompressionAction) -> ActionResult:
        s = self._state
        layer_name = action.target_layer
        target_bits = action.target_bits

        # Validate layer exists
        ls = self._find_layer_status(layer_name)
        if ls is None:
            return ActionResult(
                applied=False, rejection_code="LAYER_NOT_FOUND",
                rejection_detail=f"Layer '{layer_name}' not found.",
            )

        # Check compressible
        module = self._modules_map.get(layer_name)
        if module is None or not _is_compressible(module):
            return ActionResult(
                applied=False, rejection_code="LAYER_NOT_COMPRESSIBLE",
                rejection_detail=f"Layer '{layer_name}' is not compressible.",
            )

        # Check frozen
        if ls.is_frozen:
            return ActionResult(
                applied=False, rejection_code="LAYER_FROZEN",
                rejection_detail=f"Layer '{layer_name}' is frozen.",
            )

        # Must reduce bits
        if target_bits >= ls.current_bits:
            return ActionResult(
                applied=False, rejection_code="INVALID_BITS",
                rejection_detail=f"target_bits={target_bits} must be < current_bits={ls.current_bits}.",
            )

        # Apply simulated quantization
        old_bits = ls.current_bits
        ls.current_bits = target_bits
        ls.compression_history.append(f"quantize→{target_bits}")

        # Accuracy penalty (simulated, deterministic)
        noise = _quantization_noise(
            layer_name, old_bits, target_bits, ls.param_count, self._total_params,
        )
        s.current_accuracy = max(0.0, s.current_accuracy - noise)

        # Compute deltas
        old_size = _layer_size_mb(ls.param_count, old_bits, ls.current_sparsity)
        new_size = _layer_size_mb(ls.param_count, target_bits, ls.current_sparsity)
        delta_size = new_size - old_size
        delta_acc = -noise

        return ActionResult(
            applied=True,
            delta_size_mb=delta_size,
            delta_accuracy=delta_acc,
            debug_summary=f"Quantized {layer_name}: {old_bits}→{target_bits}bit, "
                          f"saved {abs(delta_size):.4f}MB, lost {noise:.2f}% acc",
        )

    def _apply_prune(self, action: CompressionAction) -> ActionResult:
        s = self._state
        layer_name = action.target_layer
        target_sparsity = action.sparsity

        ls = self._find_layer_status(layer_name)
        if ls is None:
            return ActionResult(
                applied=False, rejection_code="LAYER_NOT_FOUND",
                rejection_detail=f"Layer '{layer_name}' not found.",
            )

        module = self._modules_map.get(layer_name)
        if module is None or not _is_compressible(module):
            return ActionResult(
                applied=False, rejection_code="LAYER_NOT_COMPRESSIBLE",
                rejection_detail=f"Layer '{layer_name}' is not compressible.",
            )

        if ls.is_frozen:
            return ActionResult(
                applied=False, rejection_code="LAYER_FROZEN",
                rejection_detail=f"Layer '{layer_name}' is frozen.",
            )

        if target_sparsity <= ls.current_sparsity:
            return ActionResult(
                applied=False, rejection_code="SPARSITY_NOT_INCREASED",
                rejection_detail=f"target_sparsity={target_sparsity:.2f} must be > "
                                 f"current_sparsity={ls.current_sparsity:.2f}.",
            )

        if not hasattr(module, 'weight'):
            return ActionResult(
                applied=False, rejection_code="NO_WEIGHT",
                rejection_detail=f"Layer '{layer_name}' has no weight parameter.",
            )

        # Snapshot accuracy before
        acc_before = s.current_accuracy

        # Apply real L1 unstructured pruning
        try:
            prune.l1_unstructured(module, name='weight', amount=target_sparsity)
            prune.remove(module, 'weight')  # make permanent
        except Exception as e:
            return ActionResult(
                applied=False, rejection_code="PRUNE_FAILED",
                rejection_detail=f"Pruning failed: {str(e)}",
            )

        # Measure actual sparsity
        weight = module.weight.data
        actual_sparsity = float((weight == 0).sum().item()) / max(weight.numel(), 1)

        old_sparsity = ls.current_sparsity
        ls.current_sparsity = actual_sparsity
        ls.compression_history.append(f"prune→{actual_sparsity:.2f}")

        # Re-evaluate accuracy (real forward pass)
        new_accuracy = evaluate_accuracy(self._model)
        s.current_accuracy = new_accuracy

        old_size = _layer_size_mb(ls.param_count, ls.current_bits, old_sparsity)
        new_size = _layer_size_mb(ls.param_count, ls.current_bits, actual_sparsity)
        delta_size = new_size - old_size
        delta_acc = new_accuracy - acc_before

        return ActionResult(
            applied=True,
            delta_size_mb=delta_size,
            delta_accuracy=delta_acc,
            debug_summary=f"Pruned {layer_name}: sparsity {old_sparsity:.2f}→{actual_sparsity:.2f}, "
                          f"Δsize={delta_size:.4f}MB, Δacc={delta_acc:.2f}%",
        )

    def _apply_fuse(self, action: CompressionAction) -> ActionResult:
        s = self._state
        group_id = action.fuse_group_id

        fg = self._find_fuse_group(group_id)
        if fg is None:
            return ActionResult(
                applied=False, rejection_code="GROUP_NOT_FOUND",
                rejection_detail=f"Fuse group '{group_id}' not found.",
            )

        if fg.is_fused:
            return ActionResult(
                applied=False, rejection_code="GROUP_ALREADY_FUSED",
                rejection_detail=f"Fuse group '{group_id}' is already fused.",
            )

        # Calculate saving from removing BN params
        saving_mb = 0.0
        for lname in fg.layer_names:
            module = self._modules_map.get(lname)
            if module is not None and isinstance(module, nn.BatchNorm2d):
                bn_params = _layer_param_count(module)
                ls = self._find_layer_status(lname)
                bits = ls.current_bits if ls else 32
                saving_mb += _layer_size_mb(bn_params, bits, 0.0)
                # Mark BN layer as fully compressed
                if ls:
                    ls.current_sparsity = 1.0  # effectively removed
                    ls.compression_history.append("fused_out")

        fg.is_fused = True

        # Fusion is mathematically equivalent → no accuracy change
        return ActionResult(
            applied=True,
            delta_size_mb=-saving_mb,
            delta_accuracy=0.0,
            debug_summary=f"Fused {group_id} ({fg.pattern}): saved {saving_mb:.4f}MB, no acc change",
        )

    def _apply_finish(self, action: CompressionAction) -> ActionResult:
        s = self._state
        accuracy_drop = s.baseline_accuracy - s.current_accuracy
        compression_ratio = s.baseline_mb / max(s.current_mb, 0.001)
        target_met = (
            compression_ratio >= s.target_compression
            and accuracy_drop <= s.max_accuracy_drop
        )

        # Check if any meaningful compression was done
        if s.step_count <= 1 and abs(s.current_mb - s.baseline_mb) < 1e-6:
            s.is_done = True
            s.game_over_reason = "agent_finished"
            return ActionResult(
                applied=True, delta_size_mb=0.0, delta_accuracy=0.0,
                debug_summary="Finished with no compression (NOOP penalty applies).",
            )

        s.is_done = True
        s.game_over_reason = "target_met" if target_met else "agent_finished"
        return ActionResult(
            applied=True, delta_size_mb=0.0, delta_accuracy=0.0,
            debug_summary=f"Agent finished. Compression={s.baseline_mb / s.current_mb:.2f}×, "
                          f"AccDrop={s.baseline_accuracy - s.current_accuracy:.2f}%",
        )

    # ═══════════════════════════════════════════════════════════════════
    # Reward computation
    # ═══════════════════════════════════════════════════════════════════

    def _compute_step_reward(
        self, action: CompressionAction, result: ActionResult,
        delta_size: float, delta_acc: float, reward_context: Dict[str, object],
    ) -> float:
        s = self._state
        layer_name = action.target_layer or ""
        ls = self._find_layer_status(layer_name)
        compression_history = reward_context.get("compression_history", [])

        target_mb = s.baseline_mb / s.target_compression
        accuracy_drop = s.baseline_accuracy - s.current_accuracy
        compression_ratio = s.baseline_mb / max(s.current_mb, 0.001)
        steps_remaining = s.max_steps - s.step_count

        # Build layer_info for information-theoretic reward fn
        layer_info = reward_context.get("layer_info")
        if layer_info is not None and ls is not None:
            layer_info = dict(layer_info)
            layer_info["new_bits"] = ls.current_bits
            layer_info["new_sparsity"] = ls.current_sparsity

        return self._reward_fn.compute_step_reward(
            delta_size_mb=result.delta_size_mb,
            delta_accuracy=result.delta_accuracy,
            baseline_mb=s.baseline_mb,
            current_mb=s.current_mb,
            target_mb=target_mb,
            accuracy_drop=accuracy_drop,
            max_accuracy_drop=s.max_accuracy_drop,
            compression_ratio=compression_ratio,
            target_compression=s.target_compression,
            steps_remaining=steps_remaining,
            max_steps=s.max_steps,
            compression_history=compression_history,
            action_applied=result.applied,
            action_type=action.action_type,
            layer_info=layer_info,
        )

    def _compute_finish_reward(self, result: ActionResult) -> float:
        s = self._state
        accuracy_drop = s.baseline_accuracy - s.current_accuracy
        compression_ratio = s.baseline_mb / max(s.current_mb, 0.001)
        steps_remaining = s.max_steps - s.step_count
        target_met = compression_ratio >= s.target_compression and accuracy_drop <= s.max_accuracy_drop

        # NOOP finish penalty
        if s.step_count <= 1 and abs(s.current_mb - s.baseline_mb) < 1e-6:
            return NOOP_FINISH_PENALTY

        return self._reward_fn.compute_finish_bonus(
            compression_ratio=compression_ratio,
            target_compression=s.target_compression,
            accuracy_drop=accuracy_drop,
            max_accuracy_drop=s.max_accuracy_drop,
            steps_remaining=steps_remaining,
            max_steps=s.max_steps,
            target_met=target_met,
        )

    def _compute_auto_finish_bonus(self) -> float:
        """Finish bonus when steps are exhausted (auto-termination)."""
        return self._compute_finish_reward(ActionResult(applied=True))

    # ═══════════════════════════════════════════════════════════════════
    # State building helpers
    # ═══════════════════════════════════════════════════════════════════

    def _build_layer_statuses(self) -> List[LayerStatus]:
        """Inspect model and create LayerStatus for every compressible leaf."""
        statuses = []
        for name, module in self._model.named_modules():
            if len(list(module.children())) > 0:
                continue  # skip containers
            param_count = _layer_param_count(module)
            if param_count == 0:
                continue  # skip param-free layers

            statuses.append(LayerStatus(
                name=name,
                layer_type=type(module).__name__,
                param_count=param_count,
                original_bits=32,
                current_bits=32,
                current_sparsity=0.0,
                is_frozen=False,
                compression_history=[],
            ))
        return statuses

    def _build_observation(
        self, step_reward: float, action_result: ActionResult,
    ) -> CompressionObservation:
        s = self._state
        accuracy_drop = s.baseline_accuracy - s.current_accuracy
        compression_ratio = s.baseline_mb / max(s.current_mb, 0.001)
        target_mb = s.baseline_mb / s.target_compression
        steps_remaining = s.max_steps - s.step_count

        # Build layer infos
        layers = []
        for ls in s.layer_status:
            module = self._modules_map.get(ls.name)
            layers.append(LayerInfo(
                name=ls.name,
                layer_type=ls.layer_type,
                param_count=ls.param_count,
                size_mb=_layer_size_mb(ls.param_count, ls.current_bits, ls.current_sparsity),
                current_bits=ls.current_bits,
                current_sparsity=ls.current_sparsity,
                is_frozen=ls.is_frozen,
                compressible=_is_compressible(module) if module else False,
            ))

        # Build fuse group infos
        fusable_groups = []
        for fg in s.fuse_groups:
            saving = self._estimate_fuse_saving(fg.layer_names, s.layer_status)
            fusable_groups.append(FuseGroup(
                group_id=fg.group_id,
                layer_names=fg.layer_names,
                pattern=fg.pattern,
                estimated_saving_mb=saving,
                is_fused=fg.is_fused,
            ))

        # Valid action types
        valid_actions = ["quantize", "prune", "finish"]
        has_unfused = any(not fg.is_fused for fg in s.fuse_groups)
        if has_unfused:
            valid_actions.insert(2, "fuse")

        return CompressionObservation(
            # Base Observation fields
            done=s.is_done,
            reward=step_reward,
            # Static episode context
            model_name=s.model_name,
            max_steps=s.max_steps,
            baseline_mb=s.baseline_mb,
            baseline_accuracy=s.baseline_accuracy,
            target_compression=s.target_compression,
            max_accuracy_drop=s.max_accuracy_drop,
            # Dynamic state
            step=s.step_count,
            steps_remaining=max(steps_remaining, 0),
            current_mb=s.current_mb,
            current_accuracy=s.current_accuracy,
            # Pre-calculated helpers
            compression_ratio=max(compression_ratio, 1.0),
            accuracy_drop=max(accuracy_drop, 0.0),
            accuracy_budget=s.max_accuracy_drop - accuracy_drop,
            size_budget_mb=s.current_mb - target_mb,
            # Layer menu
            layers=layers,
            fusable_groups=fusable_groups,
            # Feedback
            last_action_result=action_result,
            valid_action_types=valid_actions,
            game_over_reason=s.game_over_reason,
        )

    def _find_layer_status(self, name: str) -> Optional[LayerStatus]:
        if self._state is None:
            return None
        for ls in self._state.layer_status:
            if ls.name == name:
                return ls
        return None

    def _find_fuse_group(self, group_id: str) -> Optional[FuseGroupStatus]:
        if self._state is None:
            return None
        for fg in self._state.fuse_groups:
            if fg.group_id == group_id:
                return fg
        return None

    def _estimate_fuse_saving(
        self, layer_names: List[str], layer_statuses: List[LayerStatus],
    ) -> float:
        """Estimate size saving from fusing: sum BN layer sizes in the group."""
        saving = 0.0
        status_map = {ls.name: ls for ls in layer_statuses}
        for name in layer_names:
            module = self._modules_map.get(name)
            if module is not None and isinstance(module, nn.BatchNorm2d):
                ls = status_map.get(name)
                if ls:
                    saving += _layer_size_mb(ls.param_count, ls.current_bits, ls.current_sparsity)
        return saving

    def _snapshot_reward_context(self, action: CompressionAction) -> Dict[str, object]:
        """Capture the pre-action values needed by reward functions."""
        context: Dict[str, object] = {
            "compression_history": [],
            "layer_info": None,
        }

        if action.target_layer:
            ls = self._find_layer_status(action.target_layer)
            if ls is not None:
                context["compression_history"] = list(ls.compression_history)
                context["layer_info"] = {
                    "prev_bits": ls.current_bits,
                    "new_bits": ls.current_bits,
                    "param_count": ls.param_count,
                    "prev_sparsity": ls.current_sparsity,
                    "new_sparsity": ls.current_sparsity,
                    "total_model_bits": self._total_params * 32,
                    "fused_params_removed": 0,
                }
            return context

        if action.action_type == "fuse" and action.fuse_group_id:
            fg = self._find_fuse_group(action.fuse_group_id)
            fused_params_removed = 0
            if fg is not None:
                for layer_name in fg.layer_names:
                    module = self._modules_map.get(layer_name)
                    if isinstance(module, nn.BatchNorm2d):
                        fused_params_removed += _layer_param_count(module)
            context["layer_info"] = {
                "prev_bits": 32,
                "new_bits": 32,
                "param_count": 0,
                "prev_sparsity": 0.0,
                "new_sparsity": 0.0,
                "total_model_bits": self._total_params * 32,
                "fused_params_removed": fused_params_removed,
            }

        return context
