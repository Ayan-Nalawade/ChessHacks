import random
import time

from chess import Move
import torch

from .utils import chess_manager, GameContext
from .model import load_policy_model, select_move_with_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = load_policy_model(device=DEVICE)

if MODEL is not None:
    print("[bot] Using trained policy model with search for move selection.")
else:
    print("[bot] No trained model loaded, using random-move fallback.")


def _fallback_random_move(ctx: GameContext) -> tuple[Move, dict[Move, float]]:
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        return None, {}

    # Make it obvious in logs if we ever hit the fallback path.
    print("[bot] WARNING: Falling back to random legal move.")

    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    move_probs = {
        move: weight / total_weight for move, weight in zip(legal_moves, move_weights)
    }
    chosen_move = random.choices(legal_moves, weights=move_weights, k=1)[0]
    return chosen_move, move_probs


def choose_move(ctx: GameContext) -> tuple[Move, dict[Move, float]]:
    if MODEL is not None:
        move, move_probs = select_move_with_model(
            ctx.board,
            MODEL,
            device=DEVICE,
            time_left_ms=ctx.timeLeft,
        )
        return move, move_probs

    return _fallback_random_move(ctx)


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Thinking about a move...")
    print(ctx.board.move_stack)
    time.sleep(0.05)

    move, move_probs = choose_move(ctx)
    if move is None:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (probably checkmated)")

    ctx.logProbabilities(move_probs)
    return move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
