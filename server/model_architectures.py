"""
PyTorch model definitions for the Edge-AI Compression Gym.

These runtime architectures mirror the training scripts under ``training/`` so
the saved checkpoints can be loaded directly by the environment.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
)

NUM_CLASSES = 10

IMAGE_SIZE = 32
PATCH_SIZE = 4
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
MLP_DIM = 256
DROPOUT = 0.1


class MiniCNN(nn.Module):
    """CNN used for the easy task."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    """Residual block used by MiniResNet."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class MiniResNet(nn.Module):
    """ResNet-style model used for the medium task."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    @staticmethod
    def _make_layer(
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.classifier(x.flatten(1))


class PatchEmbedding(nn.Module):
    """Patch projection for the vision transformer."""

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = PATCH_SIZE,
        d_model: int = D_MODEL,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block used by MiniTransformer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class MiniTransformer(nn.Module):
    """Vision transformer used for the hard task."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
        self.patch_embed = PatchEmbedding()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, D_MODEL))
        self.pos_drop = nn.Dropout(DROPOUT)
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(D_MODEL, N_HEADS, MLP_DIM, DROPOUT)
                for _ in range(N_LAYERS)
            ]
        )
        self.norm = nn.LayerNorm(D_MODEL)
        self.classifier = nn.Linear(D_MODEL, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        x = self.norm(x)
        return self.classifier(x[:, 0])


MODEL_REGISTRY = {
    "cnn": {"class": MiniCNN, "file": "mini_cnn.pth", "name": "MiniCNN"},
    "resnet": {"class": MiniResNet, "file": "mini_resnet.pth", "name": "MiniResNet"},
    "transformer": {
        "class": MiniTransformer,
        "file": "mini_transformer.pth",
        "name": "MiniTransformer",
    },
}


def load_model(model_key: str) -> tuple[nn.Module, float, str]:
    """Load a pre-trained model by key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_key '{model_key}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    info = MODEL_REGISTRY[model_key]
    model = info["class"]()

    checkpoint_path = os.path.join(ASSETS_DIR, info["file"])
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    baseline_accuracy = float(checkpoint.get("best_acc", 0.0))
    return model, baseline_accuracy, info["name"]
