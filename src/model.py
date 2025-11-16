import os
from typing import Iterable, List, Tuple, Dict, Optional

import chess
import torch
from torch import nn
import torch.nn.functional as F


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Encode a python-chess Board into a 1D float tensor.

    The encoding is intentionally simple and lightweight:
    - 12 piece planes (6 per color) over 8x8 squares
    - side to move
    - castling rights (4 bits)
    - en passant file (normalized 0-1, or 0 if none)
    - normalized fullmove number
    """
    planes = torch.zeros(12, 8, 8, dtype=torch.float32)

    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type  # 1..6
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane_idx = color_offset + (piece_type - 1)
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        planes[plane_idx, rank, file] = 1.0

    flat = planes.view(-1)

    side_to_move = torch.tensor(
        [1.0 if board.turn == chess.WHITE else 0.0],
        dtype=torch.float32,
    )

    castling = torch.tensor(
        [
            float(board.has_kingside_castling_rights(chess.WHITE)),
            float(board.has_queenside_castling_rights(chess.WHITE)),
            float(board.has_kingside_castling_rights(chess.BLACK)),
            float(board.has_queenside_castling_rights(chess.BLACK)),
        ],
        dtype=torch.float32,
    )

    if board.ep_square is not None:
        ep_file_norm = chess.square_file(board.ep_square) / 7.0
    else:
        ep_file_norm = 0.0
    ep = torch.tensor([ep_file_norm], dtype=torch.float32)

    move_count = torch.tensor([min(board.fullmove_number, 200) / 200.0], dtype=torch.float32)

    return torch.cat([flat, side_to_move, castling, ep, move_count], dim=0)


def encode_move(move: chess.Move, board: chess.Board) -> torch.Tensor:
    """
    Encode a single move into a small feature vector.

    Features:
    - from square rank and file (normalized 0-1)
    - to square rank and file (normalized 0-1)
    - is capture flag
    - is promotion flag
    - promotion type one-hot (Q, R, B, N)
    """
    from_rank = chess.square_rank(move.from_square) / 7.0
    from_file = chess.square_file(move.from_square) / 7.0
    to_rank = chess.square_rank(move.to_square) / 7.0
    to_file = chess.square_file(move.to_square) / 7.0

    is_capture = float(board.is_capture(move))
    is_promotion = float(move.promotion is not None)

    promo_vec = [0.0, 0.0, 0.0, 0.0]
    if move.promotion is not None:
        if move.promotion == chess.QUEEN:
            promo_vec[0] = 1.0
        elif move.promotion == chess.ROOK:
            promo_vec[1] = 1.0
        elif move.promotion == chess.BISHOP:
            promo_vec[2] = 1.0
        elif move.promotion == chess.KNIGHT:
            promo_vec[3] = 1.0

    return torch.tensor(
        [
            from_rank,
            from_file,
            to_rank,
            to_file,
            is_capture,
            is_promotion,
            *promo_vec,
        ],
        dtype=torch.float32,
    )


def encode_legal_moves(board: chess.Board, legal_moves: Iterable[chess.Move]) -> torch.Tensor:
    encoded_moves = [encode_move(m, board) for m in legal_moves]
    if not encoded_moves:
        return torch.empty(0, 10, dtype=torch.float32)
    return torch.stack(encoded_moves, dim=0)


class PolicyNetwork(nn.Module):
    """
    Simple feedforward network that scores (board, move) pairs.
    """

    def __init__(self, board_dim: int, move_dim: int, hidden_dim: int = 768):
        super().__init__()
        input_dim = board_dim + move_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, board_features: torch.Tensor, move_features: torch.Tensor) -> torch.Tensor:
        """
        board_features: (board_dim,) or (N, board_dim)
        move_features: (M, move_dim) or (N, move_dim)

        If board_features is 1D and move_features is 2D, the board
        features will be broadcast across all moves.
        """
        if board_features.dim() == 1 and move_features.dim() == 2:
            board_features = board_features.unsqueeze(0).expand(move_features.size(0), -1)

        x = torch.cat([board_features, move_features], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)
        return x


# --- Simple static evaluation for blunder avoidance ---

PIECE_VALUES: Dict[int, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}


def material_score(board: chess.Board) -> float:
    """
    Basic material evaluation: positive if White is ahead, negative if Black is ahead.
    """
    score = 0.0
    for piece in board.piece_map().values():
        value = PIECE_VALUES.get(piece.piece_type, 0.0)
        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value
    return score


def evaluate_for_side(board: chess.Board, side: bool) -> float:
    """
    Static eval from the perspective of `side` (True = White, False = Black).
    """
    score = material_score(board)
    return score if side == chess.WHITE else -score


def default_model_path() -> str:
    """
    Default location where we look for the trained model.
    Can be overridden with the CHESS_MODEL_PATH environment variable.
    """
    env_path = os.getenv("CHESS_MODEL_PATH")
    if env_path:
        return env_path

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "model", "policy_latest.pt")


def load_policy_model(
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Optional[PolicyNetwork]:
    if device is None:
        device = _device()

    if model_path is None:
        model_path = default_model_path()

    board_dim = encode_board(chess.Board()).numel()
    move_dim = encode_move(chess.Move.from_uci("e2e4"), chess.Board()).numel()

    model = PolicyNetwork(board_dim=board_dim, move_dim=move_dim)

    if not os.path.exists(model_path):
        print(f"[model] No trained model found at {model_path}, falling back to random play.")
        return None

    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        print(f"[model] Loaded trained model from {model_path}")
    except Exception as exc:
        print(f"[model] Failed to load model from {model_path}: {exc}")
        return None

    return model


def select_move_with_model(
    board: chess.Board,
    model: PolicyNetwork,
    device: Optional[torch.device] = None,
    time_left_ms: Optional[int] = None,
    max_root_moves: int = 10,
    max_depth: int = 3,
    node_limit: int = 20000,
    safety_margin_pawns: float = 3.0,
) -> Tuple[chess.Move, Dict[chess.Move, float]]:
    """
    Given a board and a trained model, compute a probability distribution
    over legal moves and pick a move.

    We use a small amount of static lookahead to avoid obvious blunders:
    - For each legal move, we evaluate the worst static score after the opponent's
      best reply (2-ply minimax using a material-only eval).
    - We restrict to moves whose score is within `safety_margin_pawns` of the best.
    - Among those "safe" moves, we pick the one with the highest model probability.
    """
    if device is None:
        device = _device()

    legal_moves_list: List[chess.Move] = list(board.generate_legal_moves())
    if not legal_moves_list:
        raise ValueError("No legal moves available")

    board_tensor = encode_board(board).to(device)
    move_tensor = encode_legal_moves(board, legal_moves_list).to(device)

    with torch.no_grad():
        scores = model(board_tensor, move_tensor)
        probs = F.softmax(scores, dim=0).cpu()

    move_probs: Dict[chess.Move, float] = {
        move: float(p) for move, p in zip(legal_moves_list, probs)
    }

    side_to_move = board.turn

    # Root move ordering by policy probability (highest first).
    order_indices = torch.argsort(probs, descending=True)
    max_root_moves = min(max_root_moves, len(legal_moves_list))
    candidate_indices: List[int] = order_indices[:max_root_moves].tolist()
    candidate_moves: List[chess.Move] = [legal_moves_list[i] for i in candidate_indices]

    # Alpha-beta search parameters.
    import time as _time

    start_time = _time.perf_counter()
    nodes_searched = 0

    # Simple time budget based on remaining time.
    if time_left_ms is not None:
        # Use at most 1/20th of remaining time for this move, capped.
        time_budget_sec = max(0.05, min(time_left_ms / 20.0, 3000.0) / 1000.0)
    else:
        time_budget_sec = None

    def time_exceeded() -> bool:
        if time_budget_sec is None:
            return False
        return (_time.perf_counter() - start_time) > time_budget_sec

    def eval_board(b: chess.Board) -> float:
        return evaluate_for_side(b, side_to_move)

    def search(
        b: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> float:
        nonlocal nodes_searched
        nodes_searched += 1

        if node_limit is not None and nodes_searched >= node_limit:
            return eval_board(b)

        if time_exceeded():
            return eval_board(b)

        if depth == 0 or b.is_game_over():
            # Game over handling.
            if b.is_game_over():
                outcome = b.outcome()
                if outcome is not None and outcome.winner is not None:
                    # Large win/loss score if the game is decided.
                    return 1000.0 if outcome.winner == side_to_move else -1000.0
                # Draw.
                return 0.0
            return eval_board(b)

        if maximizing:
            value = -float("inf")
            for mv in b.generate_legal_moves():
                child = b.copy(stack=False)
                child.push(mv)
                value = max(value, search(child, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float("inf")
            for mv in b.generate_legal_moves():
                child = b.copy(stack=False)
                child.push(mv)
                value = min(value, search(child, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    # Evaluate each candidate move with a shallow search.
    candidate_scores: List[float] = []
    for root_move in candidate_moves:
        root_board = board.copy(stack=False)
        root_board.push(root_move)
        score = search(
            root_board,
            depth=max_depth - 1,
            alpha=-float("inf"),
            beta=float("inf"),
            maximizing=False,
        )
        candidate_scores.append(score)

    # Convert to tensor for masking.
    scores_tensor = torch.tensor(candidate_scores)
    best_score = float(scores_tensor.max())

    # Keep moves within safety margin of the best.
    safe_mask = scores_tensor >= best_score - safety_margin_pawns
    safe_indices = torch.nonzero(safe_mask, as_tuple=False).view(-1)

    if len(safe_indices) == 0:
        # Fallback: pick the highest-probability move from the policy.
        best_root_idx = int(torch.argmax(probs).item())
        selected_move = legal_moves_list[best_root_idx]
        return selected_move, move_probs

    # Among safe candidate moves, choose the one with highest policy probability.
    best_move: Optional[chess.Move] = None
    best_prob = -1.0
    for local_idx in safe_indices.tolist():
        move = candidate_moves[local_idx]
        prob = move_probs.get(move, 0.0)
        if prob > best_prob:
            best_prob = prob
            best_move = move

    if best_move is None:
        # Should not happen, but fall back to policy argmax.
        best_root_idx = int(torch.argmax(probs).item())
        best_move = legal_moves_list[best_root_idx]

    return best_move, move_probs


__all__ = [
    "encode_board",
    "encode_move",
    "encode_legal_moves",
    "PolicyNetwork",
    "load_policy_model",
    "select_move_with_model",
    "material_score",
    "evaluate_for_side",
]
