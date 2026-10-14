"""
models.py — Action / Observation / State schemas for the Edge-AI Compression Gym.

Defines the complete API contract between the AI agent and the compression
environment server.  All three classes inherit from the OpenEnv base models
so they plug directly into the openenv FastAPI harness.

# ACCURACY CONVENTION: All accuracy values are in percentage points (0.0–100.0).
# accuracy_drop of 5.0 means 5 percentage points lost, NOT 0.05.
# Ensure all reward formulas use percentage-scale numbers consistently.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Bug 1 fixed: corrected import path from openenv.core.env_server.types
from openenv.core.env_server.types import Action, Observation, State


# ═══════════════════════════════════════════════════════════════════════════════
# Helper models  (used inside Observation and State)
# ═══════════════════════════════════════════════════════════════════════════════

class LayerInfo(BaseModel):
    """Per-layer snapshot visible to the agent in the observation."""

    name: str = Field(
        ..., description="Fully qualified layer name, e.g. 'features.0' or 'encoder.2.mlp.0'"
    )
    layer_type: str = Field(
        ..., description="PyTorch class name, e.g. 'Conv2d', 'Linear', 'MultiheadAttention'"
    )
    param_count: int = Field(
        ..., ge=0, description="Number of trainable parameters in this layer"
    )
    size_mb: float = Field(
        ..., ge=0.0, description="This layer's contribution to total model size in MB"
    )
    current_bits: int = Field(
        default=32, description="Current bit-precision (32=full FP32, 16=FP16, 8=INT8, 4=INT4)"
    )
    current_sparsity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of weights pruned (0.0=dense, 0.5=50% pruned)"
    )
    is_frozen: bool = Field(
        default=False, description="True if the agent has frozen this layer from further changes"
    )
    # Design flaw 2: compressible is kept here in LayerInfo but must be computed
    # fresh each step in environment.py using _is_compressible(module), never stored
    # as a static field. The server populates this field when building the observation.
    compressible: bool = Field(
        default=True, description="False for BatchNorm, Dropout, LayerNorm, small biases, etc. "
                                  "Always computed fresh from layer type — never hand-authored."
    )


class FuseGroup(BaseModel):
    """A set of consecutive layers eligible for fusion, visible to the agent."""

    group_id: str = Field(
        ..., description="Unique identifier, e.g. 'fuse_0', 'fuse_1'"
    )
    layer_names: List[str] = Field(
        ..., description="Ordered list of layer names in this group"
    )
    pattern: str = Field(
        ..., description="Human-readable pattern, e.g. 'Conv2d → BatchNorm2d → ReLU'"
    )
    # Design flaw 3 fixed: estimated_saving_mb is recomputed every step based on
    # current layer sizes, not stored statically from reset time.
    estimated_saving_mb: float = Field(
        ..., ge=0.0, description="Approximate size saving if fused RIGHT NOW "
                                 "(recomputed each step based on current layer sizes, "
                                 "accounts for any prior quantization)."
    )
    is_fused: bool = Field(
        default=False, description="Whether this group has already been fused"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Helper model for structured action feedback
# ═══════════════════════════════════════════════════════════════════════════════

class ActionResult(BaseModel):
    """Structured result of the last action — replaces the free-form string."""

    applied: bool = Field(
        ..., description="True if the action was accepted and applied; False if rejected"
    )
    delta_size_mb: float = Field(
        default=0.0, description="Change in model size in MB (negative = smaller). 0.0 if rejected."
    )
    delta_accuracy: float = Field(
        default=0.0, description="Change in accuracy in percentage points (negative = worse). 0.0 if rejected."
    )
    rejection_code: Optional[str] = Field(
        default=None,
        description="Machine-readable rejection reason, e.g. 'LAYER_FROZEN', 'INVALID_BITS', "
                    "'GROUP_ALREADY_FUSED', 'LAYER_NOT_COMPRESSIBLE'. None if applied=True."
    )
    rejection_detail: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of the rejection. None if applied=True."
    )
    debug_summary: Optional[str] = Field(
        default=None,
        description="Optional human-readable summary for debugging, "
                    "e.g. 'Quantized features.0 to INT8: saved 0.3 MB, lost 0.1% acc'. "
                    "Not intended for agent decision-making."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Typed action record for audit trail
# ═══════════════════════════════════════════════════════════════════════════════

class ActionRecord(BaseModel):
    """Typed log entry for a single step — replaces List[dict] in action_history."""

    step: int = Field(..., ge=0, description="Step index at which this action was taken")
    action_type: str = Field(..., description="The action type that was submitted")
    target_layer: Optional[str] = Field(default=None, description="Layer targeted, if any")
    target_bits: Optional[int] = Field(default=None, description="Bit-width requested, if any")
    sparsity: Optional[float] = Field(default=None, description="Sparsity level requested, if any")
    fuse_group_id: Optional[str] = Field(default=None, description="Fusion group ID, if any")
    result: ActionResult = Field(..., description="Structured outcome of this action")
    reward: float = Field(..., description="Step reward awarded for this action")


# ═══════════════════════════════════════════════════════════════════════════════
# ACTION — what the AI agent sends to the server
# ═══════════════════════════════════════════════════════════════════════════════

class CompressionAction(Action):
    """
    Single action class with optional fields gated by ``action_type``.

    The server validates that the correct fields are present for each type
    and rejects illegal moves (frozen layers, out-of-range values, etc.).
    Rejected actions still consume a step but yield ``reward = 0``.

    Action types
    ------------
    quantize : Reduce the bit-precision of a layer.
               Re-compression is allowed (e.g. 32 → 16 → 8 → 4).
               Requires: target_layer, target_bits.
    prune    : Zero out a fraction of weights in a layer.
               Can be applied cumulatively; sparsity is the NEW absolute
               target fraction, not an increment.
               Requires: target_layer, sparsity.
    fuse     : Merge a group of consecutive layers (e.g. Conv+BN+ReLU).
               Uses ``fuse_group_id`` from the observation's ``fusable_groups``.
               Requires: fuse_group_id.
    finish   : End the episode early and trigger final scoring.
               No extra fields allowed.

    Note: ``freeze`` has been removed from the public action space to prevent
    exploit strategies that bypass meaningful compression exploration.
    """

    # Bug 5 / review doc: action_type no longer includes "freeze"
    action_type: Literal["quantize", "prune", "fuse", "finish"] = Field(
        ..., description="The compression operation to perform"
    )

    # ── shared across quantize / prune ──────────────────────────────────
    target_layer: Optional[str] = Field(
        default=None,
        description="Layer name from the observation's layer menu. "
                    "Required for quantize and prune actions.",
    )

    # ── quantize-specific ───────────────────────────────────────────────
    # Bug 5 fixed: Literal[4, 8, 16] enforces valid values at deserialization time
    target_bits: Optional[Literal[4, 8, 16]] = Field(
        default=None,
        description="New bit-precision for quantize action. Must be 4, 8, or 16. "
                    "Invalid values are rejected by Pydantic before reaching env logic.",
    )

    # ── prune-specific ───────────────────────────────────────────────────
    sparsity: Optional[float] = Field(
        default=None,
        ge=0.05,
        le=0.95,
        description="Target absolute sparsity fraction for prune action (0.05–0.95). "
                    "This is the desired final fraction of zeroed weights, not an increment. "
                    "The server applies PyTorch l1_unstructured with amount=(requested - current) "
                    "to reach the target; actual resulting sparsity is measured post-prune.",
    )

    # ── fuse-specific ────────────────────────────────────────────────────
    fuse_group_id: Optional[str] = Field(
        default=None,
        description="ID of the fusion group from observation's fusable_groups list. "
                    "Required for fuse action.",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_action_fields(self) -> "CompressionAction":
        """
        Reject invalid field combinations before the action reaches environment logic.

        Rules
        -----
        quantize : requires target_layer and target_bits; no sparsity or fuse_group_id
        prune    : requires target_layer and sparsity; no target_bits or fuse_group_id
        fuse     : requires fuse_group_id; no target_layer, target_bits, or sparsity
        finish   : no extra fields allowed
        """
        t = self.action_type

        if t == "quantize":
            if self.target_layer is None:
                raise ValueError("quantize action requires target_layer")
            if self.target_bits is None:
                raise ValueError("quantize action requires target_bits (4, 8, or 16)")
            if self.sparsity is not None:
                raise ValueError("quantize action must not include sparsity")
            if self.fuse_group_id is not None:
                raise ValueError("quantize action must not include fuse_group_id")

        elif t == "prune":
            if self.target_layer is None:
                raise ValueError("prune action requires target_layer")
            if self.sparsity is None:
                raise ValueError("prune action requires sparsity (0.05–0.95)")
            if self.target_bits is not None:
                raise ValueError("prune action must not include target_bits")
            if self.fuse_group_id is not None:
                raise ValueError("prune action must not include fuse_group_id")

        elif t == "fuse":
            if self.fuse_group_id is None:
                raise ValueError("fuse action requires fuse_group_id")
            if self.target_layer is not None:
                raise ValueError("fuse action must not include target_layer")
            if self.target_bits is not None:
                raise ValueError("fuse action must not include target_bits")
            if self.sparsity is not None:
                raise ValueError("fuse action must not include sparsity")

        elif t == "finish":
            if self.target_layer is not None:
                raise ValueError("finish action must not include target_layer")
            if self.target_bits is not None:
                raise ValueError("finish action must not include target_bits")
            if self.sparsity is not None:
                raise ValueError("finish action must not include sparsity")
            if self.fuse_group_id is not None:
                raise ValueError("finish action must not include fuse_group_id")

        return self


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVATION — what the AI agent sees after each step
# ═══════════════════════════════════════════════════════════════════════════════

class CompressionObservation(Observation):
    """
    The agent's complete view of the world.

    All derived / convenience fields are pre-calculated by the server so the
    agent can focus purely on strategy.  Inherits ``done: bool`` and
    ``reward: Optional[float]`` from the OpenEnv base ``Observation``.

    Note: cumulative_reward has been intentionally omitted here to avoid
    drift with the framework's externally accumulated reward signal.
    Track cumulative reward in CompressionState only.
    """

    # ── episode context (static — set at reset, never changes) ──────────
    model_name: str = Field(
        ..., description="Architecture name: 'MiniCNN', 'MiniResNet', or 'MiniTransformer'"
    )
    max_steps: int = Field(
        ..., gt=0, description="Total actions allowed in this episode"
    )
    baseline_mb: float = Field(
        ..., gt=0.0, description="Original model file size in MB (before any compression)"
    )
    baseline_accuracy: float = Field(
        ..., ge=0.0, le=100.0, description="Original validation accuracy as a percentage"
    )
    target_compression: float = Field(
        ..., gt=1.0, description="Goal compression ratio (e.g. 2.0 = model must be ≤ half its original size)"
    )
    # Design flaw 4 fixed: expose max_accuracy_drop so the agent knows the hard ceiling
    max_accuracy_drop: float = Field(
        ..., gt=0.0,
        description="Hard ceiling for accuracy loss in percentage points. "
                    "Exceeding this triggers 'accuracy_blown' game-over immediately."
    )

    # ── dynamic state (changes every step) ──────────────────────────────
    step: int = Field(
        ..., ge=0, description="Current step number (0 at reset, 1 after first action)"
    )
    steps_remaining: int = Field(
        ..., ge=0, description="Actions left before the episode auto-terminates"
    )
    current_mb: float = Field(
        ..., ge=0.0, description="Live model file size in MB right now"
    )
    current_accuracy: float = Field(
        ..., ge=0.0, le=100.0, description="Live validation accuracy right now as a percentage"
    )

    # ── pre-calculated helpers (so the agent never does math) ───────────
    compression_ratio: float = Field(
        ..., ge=1.0, description="baseline_mb / current_mb — higher means more compressed"
    )
    accuracy_drop: float = Field(
        ..., ge=0.0, description="baseline_accuracy - current_accuracy — always ≥ 0"
    )
    accuracy_budget: float = Field(
        ..., description="Remaining accuracy the agent can afford to lose before game-over. "
                         "Equals max_accuracy_drop - accuracy_drop."
    )
    size_budget_mb: float = Field(
        ..., description="MB still to shed to hit the target. "
                         "current_mb - (baseline_mb / target_compression). "
                         "≤ 0 means the target is already met."
    )

    # Bug 2 fixed: cumulative_reward removed from Observation.
    # It lives only in CompressionState to avoid drift with the framework's
    # externally accumulated obs.reward signal.

    # ── layer menu & fusion groups ──────────────────────────────────────
    layers: List[LayerInfo] = Field(
        default_factory=list, description="Every layer in the model with its current status"
    )
    fusable_groups: List[FuseGroup] = Field(
        default_factory=list, description="Server-identified groups eligible for layer fusion"
    )

    # ── feedback from last action (structured) ──────────────────────────
    last_action_result: ActionResult = Field(
        default_factory=lambda: ActionResult(applied=False),
        description="Structured outcome of the last action. "
                    "Use applied, delta_size_mb, delta_accuracy, rejection_code for decisions. "
                    "debug_summary is human-readable convenience only.",
    )

    # ── valid actions hint ───────────────────────────────────────────────
    # Missing thing fixed: expose which action types are currently legal
    valid_action_types: List[str] = Field(
        default_factory=list,
        description="Action types that are currently legal given the environment state. "
                    "'fuse' is absent when all fusable groups are already fused. "
                    "'finish' is always present. "
                    "Sending an action type not listed here will be rejected with reward=0."
    )

    # ── episode termination info ────────────────────────────────────────
    game_over_reason: Optional[str] = Field(
        default=None,
        description="None while playing. When done=True, one of: "
                    "'target_met' (success), 'agent_finished' (agent called finish), "
                    "'steps_exhausted' (out of steps), 'accuracy_blown' (exceeded accuracy budget)",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STATE — server's hidden ledger (agent never sees this)
# ═══════════════════════════════════════════════════════════════════════════════

class LayerStatus(BaseModel):
    """Server's complete per-layer ground truth for validation and reward."""

    name: str = Field(..., description="Fully qualified layer name")
    layer_type: str = Field(..., description="PyTorch class name")
    param_count: int = Field(..., ge=0, description="Number of trainable parameters")
    original_bits: int = Field(
        default=32, description="Starting precision (always 32 for FP32)"
    )
    current_bits: int = Field(
        default=32, description="Current precision after quantization(s)"
    )
    # Bug 4 fixed: description clarifies this is measured post-prune, not a running sum
    current_sparsity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Actual fraction of zeroed weights measured post-prune, "
                    "not the cumulative sum of sparsity action values. "
                    "Must be computed from actual zero-weight counts after each prune operation."
    )
    # Design flaw 2 fixed: compressible is NOT stored here to avoid drift with LayerInfo.
    # The server computes it fresh each step using _is_compressible(module) in environment.py.
    # NON_COMPRESSIBLE_TYPES = {'BatchNorm1d', 'BatchNorm2d', 'Dropout', 'LayerNorm'}
    is_frozen: bool = Field(
        default=False, description="True if the agent has called freeze on this layer"
    )
    compression_history: List[str] = Field(
        default_factory=list,
        description="Ordered log of every operation applied to this layer. "
                    "e.g. ['quantize→16', 'prune→0.3', 'quantize→8']. "
                    "Used by the reward function to detect and score re-compression.",
    )


class FuseGroupStatus(BaseModel):
    """Server's internal tracking for a fusion group."""

    group_id: str = Field(..., description="Matches the group_id exposed in observation")
    layer_names: List[str] = Field(..., description="Layers in this fusion group")
    pattern: str = Field(..., description="Human-readable fusion pattern")
    is_fused: bool = Field(default=False, description="Whether this group has been fused")
    # Design flaw 3 fixed: recomputed each step, not stored statically from reset
    estimated_saving_mb: float = Field(
        ..., ge=0.0,
        description="Approximate size saving if fused RIGHT NOW "
                    "(recomputed each step based on current layer sizes, "
                    "accounts for any prior quantization — not a static reset-time estimate)."
    )


class CompressionState(State):
    """
    The server's absolute source of truth for the running episode.

    The agent never sees this directly.  The server uses it to validate
    actions, compute rewards, enforce limits, and generate the
    ``CompressionObservation`` that the agent receives.

    Inherits ``episode_id: Optional[str]`` and ``step_count: int``
    from the OpenEnv base ``State``.
    """

    # ── episode configuration (set at reset, immutable) ─────────────────
    task_id: str = Field(
        ..., description="Difficulty level: 'easy', 'medium', or 'hard'"
    )
    model_name: str = Field(
        ..., description="Architecture name: 'MiniCNN', 'MiniResNet', or 'MiniTransformer'"
    )
    # Design flaw 1 fixed: model_path replaced with model_key to avoid leaking
    # filesystem layout via the /state endpoint and to survive container path changes.
    # The server resolves this to an absolute path internally:
    #   ASSETS_DIR / f"mini_{model_key}.pth"
    model_key: str = Field(
        ..., description="Model identifier, one of: 'cnn', 'resnet', 'transformer'. "
                         "The server resolves this to the actual .pth path internally."
    )
    max_steps: int = Field(
        ..., gt=0, description="Action limit for this episode"
    )
    target_compression: float = Field(
        ..., gt=1.0, description="Required compression ratio to win"
    )
    max_accuracy_drop: float = Field(
        ..., gt=0.0, description="Hard ceiling for accuracy loss in percentage points"
    )

    # ── baselines (frozen at reset, never change) ───────────────────────
    baseline_mb: float = Field(
        ..., gt=0.0, description="Original model file size in MB"
    )
    baseline_accuracy: float = Field(
        ..., ge=0.0, le=100.0, description="Original validation accuracy as a percentage"
    )

    # ── live tracking ───────────────────────────────────────────────────
    current_mb: float = Field(
        ..., ge=0.0, description="Running model file size after compressions"
    )
    current_accuracy: float = Field(
        ..., ge=0.0, le=100.0, description="Running validation accuracy after compressions"
    )
    cumulative_reward: float = Field(
        default=0.0, description="Running total score (tracked in State only, not in Observation)"
    )
    is_done: bool = Field(
        default=False, description="Whether the episode has ended"
    )
    game_over_reason: Optional[str] = Field(
        default=None,
        description="Reason for termination when is_done=True",
    )

    # ── per-layer ground truth ──────────────────────────────────────────
    layer_status: List[LayerStatus] = Field(
        default_factory=list, description="Complete status of every layer in the model"
    )
    fuse_groups: List[FuseGroupStatus] = Field(
        default_factory=list, description="Tracking for all identified fusion groups"
    )

    # ── audit trail (typed, not raw dicts) ──────────────────────────────
    # Review doc fix: action_history is now a typed List[ActionRecord] instead of List[dict]
    action_history: List[ActionRecord] = Field(
        default_factory=list,
        description="Typed log of every action and its result. "
                    "Supports debugging, replay, grading, and plotting.",
    )
