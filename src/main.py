from .utils import chess_manager, GameContext
from chess import Move, WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
import random
import time
import json
import os
from huggingface_hub import hf_hub_download

# Piece values
PIECE_VALUES = {
    PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 0
}

# Load the model from Hugging Face
LOADED_MODEL = {}
try:
    repo_id = "AN707/ChessHacks"
    model_filename = "model.json"
    downloaded_model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, repo_type="dataset")
    with open(downloaded_model_path, "r") as f:
        LOADED_MODEL = json.load(f)
    print(f"Model loaded from Hugging Face: {downloaded_model_path}")
except Exception as e:
    print(f"Could not load model from Hugging Face: {e}")
    print("Proceeding without a pre-trained model.")

def evaluate_board(board):
    """
    Evaluates the board based on material.
    Positive score for White, negative for Black.
    """
    score = 0
    for piece_type in PIECE_VALUES:
        score += PIECE_VALUES[piece_type] * (
            len(board.pieces(piece_type, WHITE)) - len(board.pieces(piece_type, BLACK))
        )
    return score

def negamax(board, depth):
    if depth == 0 or board.is_game_over():
        # Evaluate from the perspective of the current player
        return evaluate_board(board) * (1 if board.turn == WHITE else -1)

    max_eval = -float('inf')
    for move in board.legal_moves:
        board.push(move)
        eval = -negamax(board, depth - 1) # Negate the result from the opponent's perspective
        board.pop()
        max_eval = max(max_eval, eval)
    return max_eval

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    print("Cooking move with negamax and variety...")
    
    # Check if the current board position is in the loaded model
    current_fen = ctx.board.fen()
    if current_fen in LOADED_MODEL:
        model_move_uci = LOADED_MODEL[current_fen]
        try:
            model_move = Move.from_uci(model_move_uci)
            if model_move in ctx.board.legal_moves:
                print(f"Move found in model: {model_move_uci}")
                return model_move
            else:
                print(f"Model move {model_move_uci} is illegal. Falling back to negamax.")
        except ValueError:
            print(f"Invalid UCI move in model: {model_move_uci}. Falling back to negamax.")

    best_eval = -float('inf')
    candidate_moves = [] # Store (move, eval) tuples
    
    search_depth = 4 

    for move in ctx.board.legal_moves:
        ctx.board.push(move)
        eval = -negamax(ctx.board, search_depth - 1) 
        ctx.board.pop()

        if eval > best_eval:
            best_eval = eval
            candidate_moves = [move] # Start a new list of candidates, storing only the move
        elif eval == best_eval:
            candidate_moves.append(move) # Add to existing candidates
            
    if not candidate_moves:
        # Fallback to random move if no candidates (should not happen with legal moves)
        legal_moves = list(ctx.board.generate_legal_moves())
        if not legal_moves:
            ctx.logProbabilities({})
            raise ValueError("No legal moves available (i probably lost didn't i)")
        return random.choice(legal_moves)

    # Randomly choose one of the best moves
    chosen_move = random.choice(candidate_moves)

    print(f"Best moves found with evaluation: {best_eval}. Chosen: {chosen_move.uci()}")
    return chosen_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
