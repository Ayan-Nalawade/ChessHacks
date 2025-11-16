"""
Modal training entrypoint for the chess policy network.

This file defines a Modal app that:
- Streams high-quality human games from the public Lichess database
- Trains a simple policy network to predict the played move
- Saves the trained model into a persistent Modal volume

Usage (from the project root or `my-chesshacks-bot` directory):

  modal run my-chesshacks-bot/train_modal.py::train

After training finishes, download the model to your local repo, e.g.:

  modal volume get chess-policy-models /models/policy_latest.pt my-chesshacks-bot/model/policy_latest.pt

The runtime bot (`src/main.py`) will automatically pick up the model from
`my-chesshacks-bot/model/policy_latest.pt` (or any path pointed to by
the `CHESS_MODEL_PATH` environment variable).
"""

import os
from typing import Iterator

import modal


app = modal.App("chess-policy-training")


IMAGE = (
    modal.Image.debian_slim()
    .pip_install(
        "torch==2.5.1",
        "python-chess==1.999",
        "zstandard",
        "requests",
        # Optional but avoids warnings from PyTorch when NumPy is missing
        "numpy",
    )
)

# Persistent volume for storing trained model weights.
# `from_name(..., create_if_missing=True)` works across Modal client versions.
MODELS_VOLUME = modal.Volume.from_name("chess-policy-models", create_if_missing=True)

# Default dataset: a monthly slice of the public Lichess standard database.
# You can override this with the LICHESS_PGN_URL environment variable when running.
LICHESS_STANDARD_URL = os.getenv(
    "LICHESS_PGN_URL",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2024-03.pgn.zst",
)


@app.function(
    image=IMAGE,
    timeout=60 * 60 * 4,
    volumes={"/models": MODELS_VOLUME},
    gpu="T4",
)
def train(
    num_games: int = 20000,
    learning_rate: float = 1e-3,
    min_elo: int = 1800,
) -> None:
    """
    Train the policy network on human games streamed from Lichess.

    Args:
        num_games: Number of PGN games to stream and train on.
        learning_rate: Adam optimizer learning rate.
    """
    import io
    import time

    import chess
    import chess.pgn
    import requests
    import torch
    import torch.nn.functional as F
    import zstandard

    # Local copies of the model + encoders so we don't depend on the
    # package layout of the repo inside the Modal container.

    def encode_board(board: chess.Board) -> torch.Tensor:
        import torch as _torch

        planes = _torch.zeros(12, 8, 8, dtype=_torch.float32)
        for square, piece in board.piece_map().items():
            piece_type = piece.piece_type
            color_offset = 0 if piece.color == chess.WHITE else 6
            plane_idx = color_offset + (piece_type - 1)
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            planes[plane_idx, rank, file] = 1.0

        flat = planes.view(-1)

        side_to_move = _torch.tensor(
            [1.0 if board.turn == chess.WHITE else 0.0],
            dtype=_torch.float32,
        )

        castling = _torch.tensor(
            [
                float(board.has_kingside_castling_rights(chess.WHITE)),
                float(board.has_queenside_castling_rights(chess.WHITE)),
                float(board.has_kingside_castling_rights(chess.BLACK)),
                float(board.has_queenside_castling_rights(chess.BLACK)),
            ],
            dtype=_torch.float32,
        )

        if board.ep_square is not None:
            ep_file_norm = chess.square_file(board.ep_square) / 7.0
        else:
            ep_file_norm = 0.0
        ep = _torch.tensor([ep_file_norm], dtype=_torch.float32)

        move_count = _torch.tensor(
            [min(board.fullmove_number, 200) / 200.0],
            dtype=_torch.float32,
        )

        return _torch.cat([flat, side_to_move, castling, ep, move_count], dim=0)

    def encode_move(move: chess.Move, board: chess.Board) -> torch.Tensor:
        import torch as _torch

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

        return _torch.tensor(
            [
                from_rank,
                from_file,
                to_rank,
                to_file,
                is_capture,
                is_promotion,
                *promo_vec,
            ],
            dtype=_torch.float32,
        )

    def encode_legal_moves(board: chess.Board, legal_moves):
        import torch as _torch

        encoded_moves = [encode_move(m, board) for m in legal_moves]
        if not encoded_moves:
            return _torch.empty(0, 10, dtype=_torch.float32)
        return _torch.stack(encoded_moves, dim=0)

    class PolicyNetwork(torch.nn.Module):
        def __init__(self, board_dim: int, move_dim: int, hidden_dim: int = 768):
            super().__init__()
            input_dim = board_dim + move_dim
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = torch.nn.Linear(hidden_dim, 1)

        def forward(self, board_features: torch.Tensor, move_features: torch.Tensor) -> torch.Tensor:
            if board_features.dim() == 1 and move_features.dim() == 2:
                board_features = board_features.unsqueeze(0).expand(move_features.size(0), -1)

            x = torch.cat([board_features, move_features], dim=-1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x).squeeze(-1)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")
    print(f"[train] Streaming games from: {LICHESS_STANDARD_URL}")

    def stream_games(limit: int, min_elo_threshold: int) -> Iterator[chess.pgn.Game]:
        resp = requests.get(LICHESS_STANDARD_URL, stream=True)
        resp.raise_for_status()

        dctx = zstandard.ZstdDecompressor()
        stream_reader = dctx.stream_reader(resp.raw)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

        accepted = 0
        scanned = 0

        try:
            while accepted < limit:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break

                scanned += 1
                headers = game.headers
                try:
                    w_elo = int(headers.get("WhiteElo", "0"))
                    b_elo = int(headers.get("BlackElo", "0"))
                except ValueError:
                    w_elo, b_elo = 0, 0

                if w_elo >= min_elo_threshold and b_elo >= min_elo_threshold:
                    accepted += 1
                    if accepted % 1000 == 0:
                        print(
                            f"[data] Accepted {accepted} games (scanned {scanned}, min_elo={min_elo_threshold})"
                        )
                    yield game
        except Exception as exc:
            # Network / streaming error – stop early but keep what we have.
            print(
                f"[data] Streaming interrupted after accepted={accepted}, scanned={scanned}: {exc}"
            )

    # Initialize model and optimizer
    board_dim = encode_board(chess.Board()).numel()
    move_dim = encode_move(chess.Move.from_uci("e2e4"), chess.Board()).numel()
    model = PolicyNetwork(board_dim=board_dim, move_dim=move_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    positions_seen = 0
    log_every = 2000
    checkpoint_every = 100000
    accum_steps = 16
    accum_loss = 0.0
    accum_count = 0
    last_log_time = time.time()

    for game_idx, game in enumerate(stream_games(limit=num_games, min_elo_threshold=min_elo), start=1):
        board = game.board()
        for move in game.mainline_moves():
            legal_moves = list(board.generate_legal_moves())
            if not legal_moves or move not in legal_moves:
                board.push(move)
                continue

            board_tensor = encode_board(board).to(device)
            moves_tensor = encode_legal_moves(board, legal_moves).to(device)

            scores = model(board_tensor, moves_tensor)
            target_index = legal_moves.index(move)
            target = torch.tensor([target_index], dtype=torch.long, device=device)

            loss = F.cross_entropy(scores.unsqueeze(0), target)

            # Gradient accumulation over multiple positions to reduce
            # optimizer overhead and make better use of the device.
            loss = loss / accum_steps
            loss.backward()
            accum_loss += float(loss.item()) * accum_steps
            accum_count += 1

            if accum_count % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            positions_seen += 1

            if positions_seen % log_every == 0:
                now = time.time()
                elapsed = now - last_log_time
                avg_loss = accum_loss / max(accum_count, 1)
                print(
                    f"[train] games={game_idx} positions={positions_seen} avg_loss={avg_loss:.4f} elapsed={elapsed:.1f}s"
                )
                last_log_time = now
                accum_loss = 0.0
                accum_count = 0

            if positions_seen % checkpoint_every == 0:
                torch.save(model.state_dict(), "/models/policy_latest.pt")
                print(f"[train] Checkpoint saved at positions={positions_seen}")

            board.push(move)

    save_path = "/models/policy_latest.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[train] Saved model to {save_path} in volume 'chess-policy-models'")


__all__ = ["train"]
