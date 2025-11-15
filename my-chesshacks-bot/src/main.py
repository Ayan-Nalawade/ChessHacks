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

MODEL_ENV_VAR = "CHESS_MODEL_PATH"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "value_net.pt"
_model_lock = threading.Lock()
_model: torch.nn.Module | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKMATE_SCORE = 1e4
SEARCH_DEPTH = 2
TOP_MOVE_COUNT = 6
TEMPERATURE = 0.3


def _resolve_model_path() -> Path:
    override = os.environ.get(MODEL_ENV_VAR)
    return Path(override) if override else DEFAULT_MODEL_PATH


def _load_model() -> torch.nn.Module:
    global _model
    with _model_lock:
        if _model is None:
            path = _resolve_model_path()
            if not path.exists():
                raise FileNotFoundError(
                    f"Model checkpoint not found at {path}. "
                    f"Set {MODEL_ENV_VAR} to override path."
                )
            model, _ = load_model(path, device=_device)
            model.eval()
            _model = model
    return _model


@lru_cache(maxsize=4096)
def _model_eval(fen: str) -> tuple[np.ndarray, float]:
    model = _load_model()
    planes = fen_to_planes(fen)
    tensor = torch.from_numpy(planes).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits, value = model(tensor)
        policy = torch.softmax(logits, dim=1).cpu().numpy()[0]
        value_scalar = float(value.squeeze().cpu().numpy())
    return policy, value_scalar


def _terminal_value(board: chess.Board) -> float | None:
    if board.is_checkmate():
        return -CHECKMATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    return None


def _legal_policy(board: chess.Board, policy: np.ndarray) -> Dict[Move, float]:
    scores: Dict[Move, float] = {}
    total = 0.0
    for move in board.generate_legal_moves():
        idx = move_to_index(board, move)
        if idx is None:
            continue
        prob = float(max(policy[idx], 0.0))
        scores[move] = prob
        total += prob

    if total <= 0:
        uniform = 1.0 / max(len(scores), 1)
        return {move: uniform for move in scores}

    return {move: score / total for move, score in scores.items()}


def _negamax(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    terminal = _terminal_value(board)
    if terminal is not None:
        return terminal

    if depth == 0:
        _, value = _model_eval(board.fen())
        return value * 1000.0

    policy, _ = _model_eval(board.fen())
    move_probs = _legal_policy(board, policy)
    ordered_moves = sorted(
        move_probs.items(), key=lambda kv: kv[1], reverse=True
    )[:TOP_MOVE_COUNT]

    if not ordered_moves:
        return 0.0

    best = -float("inf")
    for move, _ in ordered_moves:
        board.push(move)
        value = -_negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        if value > best:
            best = value
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    return best


def _softmax(scores: Dict[Move, float]) -> Dict[Move, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {
        move: math.exp((score - max_score) / TEMPERATURE)
        for move, score in scores.items()
    }
    total = sum(exp_scores.values())
    if total <= 0:
        uniform = 1.0 / len(scores)
        return {move: uniform for move in scores}
    return {move: val / total for move, val in exp_scores.items()}


def _select_move(ctx: GameContext) -> Move:
    board = ctx.board
    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("Position has no legal moves.")

    try:
        policy, _ = _model_eval(board.fen())
        move_probs = _legal_policy(board, policy)
        move_scores = {}
        for move in legal_moves:
            board.push(move)
            move_scores[move] = -_negamax(board, SEARCH_DEPTH - 1, -float("inf"), float("inf"))
            board.pop()
    except Exception as exc:
        print(f"Model inference failed, falling back to uniform random policy: {exc}")
        uniform = {move: 1.0 / len(legal_moves) for move in legal_moves}
        ctx.logProbabilities(uniform)
        return legal_moves[0]

    ctx.logProbabilities(move_probs)
    best_move = max(move_scores, key=move_scores.get)
    return best_move


@chess_manager.entrypoint
def model_entrypoint(ctx: GameContext) -> Move:
    return _select_move(ctx)


@chess_manager.reset
def reset_func(ctx: GameContext):
    _model_eval.cache_clear()
