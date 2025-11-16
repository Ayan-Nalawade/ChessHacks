from __future__ import annotations

import math
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import chess
from chess import Move

from .model import fen_to_planes, load_model
from .move_encoding import move_to_index
from .utils import GameContext, chess_manager


from .trihModel import EvaluationNetwork, NeuralChessEngine


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

evalModel = EvaluationNetwork(input_planes=19, num_blocks=8, channels=192)
checkpoint_path = Path(r"C:\ZJL\Michael\chesshacks\ChessHacks\my-chesshacks-bot\artifacts\trihModel.pt").expanduser().resolve()

checkpoint = torch.load(checkpoint_path, map_location=device)
if isinstance(checkpoint, torch.nn.Module):
    checkpoint = checkpoint.state_dict()
elif "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]
elif "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

evalModel.load_state_dict(checkpoint)
evalModel.to(device)
evalModel.eval()

engine = NeuralChessEngine(evalModel)




def _select_move(ctx: GameContext) -> Move:
    board = ctx.board
    move, info = engine.select_move(board)

    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("Position has no legal moves.")

    try:
       move, info = engine.select_move(board)
    except Exception as exc:
        print(f"Model inference failed, falling back to uniform random policy: {exc}")
        uniform = {move: 1.0 / len(legal_moves) for move in legal_moves}
        ctx.logProbabilities(uniform)
        return legal_moves[0]

    
    return move


@chess_manager.entrypoint
def model_entrypoint(ctx: GameContext) -> Move:
    return _select_move(ctx)


@chess_manager.reset
def reset_func(ctx: GameContext):
    _model_eval.cache_clear()
