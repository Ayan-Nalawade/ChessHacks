"""
Streaming dataset that yields (planes, policy index, value) tuples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional, Sequence

import chess
import torch
from torch.utils.data import IterableDataset

from .model import fen_to_planes
from .move_encoding import ACTION_SIZE, move_to_index


RESULT_MAP = {
    "1-0": 1.0,
    "0-1": -1.0,
    "1/2-1/2": 0.0,
}


class PolicyValueDataset(IterableDataset):
    def __init__(
        self,
        files: Sequence[Path | str],
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.files = [Path(f) for f in files]
        self.max_samples = max_samples

    def parse_line(self, line: str) -> Optional[tuple[torch.Tensor, int, float]]:
        payload = json.loads(line)
        board = chess.Board(payload["fen"])

        result = payload.get("result")
        if result not in RESULT_MAP:
            return None
        value = RESULT_MAP[result]
        if board.turn == chess.BLACK:
            value = -value

        move_uci = payload.get("uci")
        if not move_uci:
            return None

        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return None

        if not board.is_legal(move):
            return None

        move_index = move_to_index(board, move)
        if move_index is None:
            return None

        planes = torch.from_numpy(fen_to_planes(payload["fen"]))
        policy_tensor = torch.tensor(move_index, dtype=torch.long)
        value_tensor = torch.tensor(value, dtype=torch.float32)
        return planes, policy_tensor, value_tensor

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int, torch.Tensor]]:
        produced = 0
        for path in self.files:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    item = self.parse_line(line)
                    if item is None:
                        continue
                    yield item
                    produced += 1
                    if self.max_samples is not None and produced >= self.max_samples:
                        return
