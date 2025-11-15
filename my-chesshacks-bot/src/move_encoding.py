"""
Move encoding utilities inspired by AlphaZero's 8x8x73 policy representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import chess

BOARD_SIZE = 8
NUM_SQUARES = 64
SLIDING_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),   # north
    (0, -1),  # south
    (1, 0),   # east
    (-1, 0),  # west
    (1, 1),   # north-east
    (-1, 1),  # north-west
    (1, -1),  # south-east
    (-1, -1), # south-west
)
SLIDING_STEPS = 7
KNIGHT_DELTAS: Tuple[Tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
UNDERPROMO_DIRECTIONS: Tuple[int, ...] = (-1, 0, 1)  # left, forward, right relative to player
UNDERPROMO_PIECES = (chess.ROOK, chess.BISHOP, chess.KNIGHT)

SLIDING_SPACE = len(SLIDING_DIRECTIONS) * SLIDING_STEPS  # 56
KNIGHT_SPACE = len(KNIGHT_DELTAS)  # 8
UNDERPROMO_SPACE = len(UNDERPROMO_DIRECTIONS) * len(UNDERPROMO_PIECES)  # 9
ACTION_PLANE_SIZE = SLIDING_SPACE + KNIGHT_SPACE + UNDERPROMO_SPACE  # 73
ACTION_SIZE = NUM_SQUARES * ACTION_PLANE_SIZE  # 4672


def _square_to_coords(square: int) -> Tuple[int, int]:
    return chess.square_file(square), chess.square_rank(square)


def _within_board(file_idx: int, rank_idx: int) -> bool:
    return 0 <= file_idx < BOARD_SIZE and 0 <= rank_idx < BOARD_SIZE


def move_to_index(board: chess.Board, move: chess.Move) -> Optional[int]:
    """Encode a move into the 0..4671 policy index."""
    from_sq = move.from_square
    to_sq = move.to_square
    base = ACTION_PLANE_SIZE * from_sq

    from_file, from_rank = _square_to_coords(from_sq)
    to_file, to_rank = _square_to_coords(to_sq)
    df = to_file - from_file
    dr = to_rank - from_rank

    # Sliding / queen-like moves (includes king, rook, bishop, queen, pawn pushes, castling, queen promotions)
    for dir_idx, (dx, dy) in enumerate(SLIDING_DIRECTIONS):
        for step in range(1, SLIDING_STEPS + 1):
            if df == dx * step and dr == dy * step:
                return base + dir_idx * SLIDING_STEPS + (step - 1)

    # Knight moves
    for knight_idx, (dx, dy) in enumerate(KNIGHT_DELTAS):
        if df == dx and dr == dy:
            return base + SLIDING_SPACE + knight_idx

    # Underpromotions (non-queen)
    if move.promotion and move.promotion in UNDERPROMO_PIECES:
        mover_color = board.color_at(from_sq)
        if mover_color is None:
            return None
        forward = 1 if mover_color == chess.WHITE else -1
        df_rel = df if mover_color == chess.WHITE else -df
        dr_rel = dr if mover_color == chess.WHITE else -dr
        if dr_rel != 1:
            return None
        if df_rel not in UNDERPROMO_DIRECTIONS:
            return None
        dir_index = UNDERPROMO_DIRECTIONS.index(df_rel)
        piece_index = UNDERPROMO_PIECES.index(move.promotion)
        promo_base = SLIDING_SPACE + KNIGHT_SPACE
        return base + promo_base + dir_index * len(UNDERPROMO_PIECES) + piece_index

    return None


def index_to_direction_step(index: int) -> Tuple[int, int]:
    """Helper for debugging - returns (file delta, rank delta) for sliding moves."""
    dir_idx, step = divmod(index, SLIDING_STEPS)
    dx, dy = SLIDING_DIRECTIONS[dir_idx]
    return dx * (step + 1), dy * (step + 1)

