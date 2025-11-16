import chess
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import time


# ============================================================================
# Board Encoding (Planar representation)
# ============================================================================

def encode_board_planes(board: chess.Board) -> torch.Tensor:
    """
    Encode board as stacked 8×8 planes - optimized for evaluation.
    
    Planes:
    - 0-5: White pieces (P, N, B, R, Q, K)
    - 6-11: Black pieces
    - 12: Side to move (1 if white, 0 if black)
    - 13: Move count / 200
    - 14-17: Castling rights (WK, WQ, BK, BQ)
    - 18: Halfmove clock / 100
    
    Total: 19 planes
    """
    planes = torch.zeros(19, 8, 8)
    
    # Piece planes (0-11)
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        piece_type = piece.piece_type - 1  # 0-5
        plane = piece_type if piece.color == chess.WHITE else piece_type + 6
        planes[plane, rank, file] = 1.0
    
    # Side to move (plane 12)
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    
    # Move count (plane 13)
    planes[13, :, :] = min(board.fullmove_number, 200) / 200.0
    
    # Castling rights (planes 14-17)
    planes[14, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[15, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[16, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[17, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    
    # No-progress count (plane 18)
    planes[18, :, :] = min(board.halfmove_clock, 100) / 100.0
    
    return planes


# ============================================================================
# ResNet Architecture - Pure Evaluation
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class EvaluationNetwork(nn.Module):
    """
    CNN-based evaluation network for alpha-beta search.
    Outputs a scalar evaluation: positive = good for white, negative = good for black.
    
    Args:
        input_planes: Number of input planes (19)
        num_blocks: Number of residual blocks (6-10 for good balance)
        channels: Number of channels in residual tower (128-256)
    """
    
    def __init__(self, input_planes: int = 19, num_blocks: int = 8, channels: int = 192):
        super().__init__()
        
        # Initial convolution
        self.conv_input = nn.Conv2d(input_planes, channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])
        
        # Value head - outputs evaluation in centipawns
        self.value_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 19, 8, 8) board planes
        
        Returns:
            eval: (batch,) position evaluation (positive = white advantage)
        """
        # Initial conv
        out = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for block in self.residual_blocks:
            out = block(out)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = self.value_fc2(value).squeeze(-1)
        
        return value


# ============================================================================
# Alpha-Beta Search with Neural Network Evaluation
# ============================================================================

class NeuralChessEngine:
    """
    Chess engine using alpha-beta search with neural network evaluation.
    """
    
    def __init__(
        self,
        model: EvaluationNetwork,
        device: Optional[torch.device] = None,
        temperature: float = 0.7,
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.temperature = max(0.05, float(temperature))
        
        # Search statistics
        self.nodes_searched = 0
        self.cache_hits = 0
        self.eval_cache: Dict[str, float] = {}

    def reset(self):
        """Clear cached evaluations and counters."""
        self.eval_cache.clear()
        self.nodes_searched = 0
        self.cache_hits = 0
    
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate position using neural network.
        Returns value in centipawns from white's perspective.
        """
        # Check cache
        fen = board.fen()
        if fen in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[fen]
        
        # Handle terminal positions
        if board.is_checkmate():
            eval_score = -10000.0 if board.turn == chess.WHITE else 10000.0
            self.eval_cache[fen] = eval_score
            return eval_score
        
        if board.is_stalemate() or board.is_insufficient_material():
            self.eval_cache[fen] = 0.0
            return 0.0
        
        # Neural network evaluation
        with torch.no_grad():
            planes = encode_board_planes(board).unsqueeze(0).to(self.device)
            eval_score = float(self.model(planes).item())
        
        self.eval_cache[fen] = eval_score
        return eval_score
    
    def order_moves(
        self,
        board: chess.Board,
        moves: List[chess.Move],
    ) -> List[chess.Move]:
        """
        Order moves for better alpha-beta pruning.
        Uses simple heuristics: captures first, then checks.
        """
        def move_priority(move: chess.Move) -> Tuple[int, int]:
            priority = 0
            
            # Captures are high priority
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                    priority += victim.piece_type * 10 - attacker.piece_type
            
            # Promotions
            if move.promotion:
                priority += 100
            
            # Checks (expensive to compute, so do it last)
            board_copy = board.copy(stack=False)
            board_copy.push(move)
            if board_copy.is_check():
                priority += 50
            
            return (-priority, 0)  # Negative for descending sort
        
        return sorted(moves, key=move_priority)

    def _probabilities_from_evals(
        self,
        move_evals: List[Tuple[chess.Move, float]],
        turn: chess.Color,
    ) -> Dict[chess.Move, float]:
        if not move_evals:
            return {}

        adjusted: List[Tuple[chess.Move, float]] = []
        for move, eval_score in move_evals:
            score = eval_score if turn == chess.WHITE else -eval_score
            adjusted.append((move, score))

        max_score = max(score for _, score in adjusted)
        exp_scores = {
            move: math.exp((score - max_score) / self.temperature)
            for move, score in adjusted
        }
        total = sum(exp_scores.values())
        if total <= 0:
            uniform = 1.0 / len(adjusted)
            return {move: uniform for move, _ in adjusted}
        return {move: value / total for move, value in exp_scores.items()}
    
    def alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        start_time: float,
        time_limit: Optional[float],
    ) -> float:
        """
        Alpha-beta search with neural network evaluation.
        """
        self.nodes_searched += 1
        
        # Time check
        if time_limit and (time.perf_counter() - start_time) > time_limit:
            return self.evaluate_position(board)
        
        # Base case: evaluate at leaf
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board)
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self.evaluate_position(board)
        
        # Move ordering for better pruning
        ordered_moves = self.order_moves(board, legal_moves)
        
        if maximizing:
            value = float('-inf')
            for move in ordered_moves:
                child = board.copy(stack=False)
                child.push(move)
                value = max(value, self.alpha_beta(
                    child, depth - 1, alpha, beta, False, start_time, time_limit
                ))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # Beta cutoff
            return value
        else:
            value = float('inf')
            for move in ordered_moves:
                child = board.copy(stack=False)
                child.push(move)
                value = min(value, self.alpha_beta(
                    child, depth - 1, alpha, beta, True, start_time, time_limit
                ))
                beta = min(beta, value)
                if beta <= alpha:
                    break  # Alpha cutoff
            return value
    
    def select_move(
        self,
        board: chess.Board,
        max_depth: int = 3,
        time_limit_ms: Optional[int] = None,
    ) -> Tuple[chess.Move, Dict[str, any]]:
        """
        Select best move using iterative deepening alpha-beta search.
        
        Args:
            board: Current position
            max_depth: Maximum search depth
            time_limit_ms: Time limit in milliseconds
        
        Returns:
            best_move: Selected move
            info: Search statistics
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        if len(legal_moves) == 1:
            return legal_moves[0], {'depth': 0, 'nodes': 0}
        
        # Initialize
        start_time = time.perf_counter()
        time_limit = time_limit_ms / 1000.0 if time_limit_ms else None
        
        self.nodes_searched = 0
        self.cache_hits = 0
        self.eval_cache.clear()
        
        best_move = legal_moves[0]
        best_eval = float('-inf') if board.turn == chess.WHITE else float('inf')
        
        # Iterative deepening
        last_move_evals: List[Tuple[chess.Move, float]] = []
        for depth in range(1, max_depth + 1):
            if time_limit and (time.perf_counter() - start_time) > time_limit * 0.8:
                break
            
            move_evals: List[Tuple[chess.Move, float]] = []
            
            for move in legal_moves:
                child = board.copy(stack=False)
                child.push(move)
                
                # Search from opponent's perspective
                eval_score = self.alpha_beta(
                    child,
                    depth - 1,
                    float('-inf'),
                    float('inf'),
                    board.turn == chess.BLACK,  # Flip perspective
                    start_time,
                    time_limit,
                )
                
                move_evals.append((move, eval_score))
                
                # Time check
                if time_limit and (time.perf_counter() - start_time) > time_limit * 0.9:
                    break

            if move_evals:
                last_move_evals = move_evals
            
            # Sort moves by evaluation
            if board.turn == chess.WHITE:
                move_evals.sort(key=lambda x: x[1], reverse=True)
                best_move = move_evals[0][0]
                best_eval = move_evals[0][1]
            else:
                move_evals.sort(key=lambda x: x[1])
                best_move = move_evals[0][0]
                best_eval = move_evals[0][1]
            
            # Reorder legal_moves for next iteration (better move ordering)
            legal_moves = [m for m, _ in move_evals]
        
        elapsed = time.perf_counter() - start_time
        
        probabilities = self._probabilities_from_evals(last_move_evals, board.turn)

        info = {
            'depth': depth,
            'nodes': self.nodes_searched,
            'time_ms': int(elapsed * 1000),
            'nps': int(self.nodes_searched / elapsed) if elapsed > 0 else 0,
            'eval': best_eval,
            'cache_hits': self.cache_hits,
            'probabilities': probabilities,
        }
        
        return best_move, info


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create model
    model = EvaluationNetwork(input_planes=19, num_blocks=8, channels=192)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create engine
    engine = NeuralChessEngine(model)
    
    # Test position
    board = chess.Board()
    
    # Select move
    print("\nSearching for best move...")
    move, info = engine.select_move(board, max_depth=7, time_limit_ms=5000)
    
    print(f"Best move: {move}")
    print(f"Depth: {info['depth']}")
    print(f"Nodes: {info['nodes']:,}")
    print(f"Time: {info['time_ms']}ms")
    print(f"NPS: {info['nps']:,}")
    print(f"Eval: {info['eval']:.2f} centipawns")
    print(f"Cache hits: {info['cache_hits']:,}")
