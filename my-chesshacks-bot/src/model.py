"""
Lightweight policy/value network plus helpers for converting FEN strings
into neural-network input planes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .move_encoding import ACTION_SIZE

LEGACY_PREFIXES = (
    "stem.",
    "blocks.",
    "policy_head.",
    "value_head.",
)

PLANE_COUNT = 18
BOARD_SIZE = 8


def fen_to_planes(fen: str) -> np.ndarray:
    """Convert a FEN position into the 18x8x8 AlphaZero-style planes."""
    board = chess.Board(fen)
    planes = np.zeros((PLANE_COUNT, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        base = 0 if piece.color == chess.WHITE else 6
        plane = piece_to_plane[piece.piece_type] + base
        planes[plane, rank, file] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        planes[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0

    # Side to move
    if board.turn == chess.WHITE:
        planes[16, :, :] = 1.0

    # En passant
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        planes[17, ep_rank, ep_file] = 1.0

    return planes


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class PolicyValueNet(nn.Module):
    """Small residual tower that predicts move policy + scalar value."""

    def __init__(
        self,
        in_planes: int = PLANE_COUNT,
        board_size: int = BOARD_SIZE,
        policy_size: int = ACTION_SIZE,
        trunk_channels: int = 128,
        num_blocks: int = 8,
    ):
        super().__init__()
        self.board_size = board_size
        self.policy_size = policy_size

        self.conv_in = nn.Conv2d(in_planes, trunk_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(trunk_channels)
        self.res_blocks = nn.ModuleList([ResidualBlock(trunk_channels) for _ in range(num_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(trunk_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, policy_size)

        # Value head
        self.value_conv = nn.Conv2d(trunk_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)

        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy_logits, value


def _convert_legacy_state_dict(state_dict: dict) -> tuple[dict, int | None]:
    if not any(any(key.startswith(prefix) for prefix in LEGACY_PREFIXES) for key in state_dict):
        return state_dict, None

    converted = {}
    max_block_index = -1

    for key, tensor in state_dict.items():
        new_key = None
        if key.startswith("stem.0."):
            new_key = key.replace("stem.0.", "conv_in.")
        elif key.startswith("stem.1."):
            new_key = key.replace("stem.1.", "bn_in.")
        elif key.startswith("blocks."):
            parts = key.split(".", 2)
            if len(parts) == 3:
                idx = int(parts[1])
                max_block_index = max(max_block_index, idx)
                new_key = f"res_blocks.{idx}.{parts[2]}"
        elif key.startswith("policy_head.0."):
            new_key = key.replace("policy_head.0.", "policy_conv.")
        elif key.startswith("policy_head.1."):
            new_key = key.replace("policy_head.1.", "policy_bn.")
        elif key.startswith("policy_head.4."):
            new_key = key.replace("policy_head.4.", "policy_fc.")
        elif key.startswith("value_head.0."):
            new_key = key.replace("value_head.0.", "value_conv.")
        elif key.startswith("value_head.1."):
            new_key = key.replace("value_head.1.", "value_bn.")
        elif key.startswith("value_head.4."):
            new_key = key.replace("value_head.4.", "value_fc1.")
        elif key.startswith("value_head.6."):
            new_key = key.replace("value_head.6.", "value_fc2.")

        if new_key is None:
            new_key = key

        converted[new_key] = tensor

    block_count = max_block_index + 1 if max_block_index >= 0 else None
    return converted, block_count


def _infer_block_count(state_dict: dict) -> int | None:
    max_idx = -1
    prefix = "res_blocks."
    for key in state_dict.keys():
        if key.startswith(prefix):
            remainder = key[len(prefix):]
            try:
                idx = int(remainder.split(".", 1)[0])
            except ValueError:
                continue
            max_idx = max(max_idx, idx)
    return max_idx + 1 if max_idx >= 0 else None


def create_model(device: torch.device | None = None, *, num_blocks: int = 8) -> PolicyValueNet:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNet(num_blocks=num_blocks).to(device)
    return model


def save_model(model: nn.Module, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)
    return path


def load_model(path: Path | str, device: torch.device | None = None) -> Tuple[nn.Module, dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(path), map_location=device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    if state_dict is None:
        state_dict = checkpoint

    state_dict, legacy_block_count = _convert_legacy_state_dict(state_dict)
    block_count = _infer_block_count(state_dict)
    num_blocks = legacy_block_count or block_count or 8
    model = PolicyValueNet(num_blocks=num_blocks).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


__all__ = [
    "fen_to_planes",
    "PolicyValueNet",
    "create_model",
    "save_model",
    "load_model",
]
