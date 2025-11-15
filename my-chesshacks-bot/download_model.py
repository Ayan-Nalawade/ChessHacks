"""
Utility script for materializing a slice of the Stockfish Fishtest PGN dataset
into JSONL training samples.

The script will:
1. Download the requested subset of `.pgn.gz` archives from HF Hub
2. Parse every game and move, extracting FENs + Stockfish eval comments
3. Emit a stream of JSON lines that can be fed into downstream preprocessing

Usage:
    python download_model.py \
        --pattern "24-12-*/*/*.pgn.gz" \
        --max-games 250 \
        --output model_dataset/fishtest_24-12.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from pathlib import Path
from typing import Iterator, Optional

import chess
import chess.pgn
from huggingface_hub import snapshot_download
from tqdm import tqdm

DEFAULT_REPO_ID = "official-stockfish/fishtest_pgns"

EVAL_TOKEN_RE = re.compile(r"\{([^}]*)\}")
TIME_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)s")


def _extract_payload(comment: str) -> Optional[str]:
    if not comment:
        return None
    match = EVAL_TOKEN_RE.search(comment)
    if match:
        payload = match.group(1).strip()
        return payload or None
    # comments may already exclude braces (e.g. "-0.94/27 4.6s")
    comment = comment.strip()
    if comment:
        return comment
    return None


def parse_comment(comment: str) -> dict:
    """Extract centipawn score, depth, mate, and time from Stockfish comment."""
    payload = _extract_payload(comment)
    if not payload:
        return {}

    tokens = payload.split()
    first = tokens[0]
    evaluation: dict[str, Optional[float | int]] = {}

    if "/" in first:
        score_token, depth_token = first.split("/", 1)
    else:
        score_token, depth_token = first, None

    score_token = score_token.strip()
    if score_token:
        if score_token.upper().startswith("M"):
            try:
                evaluation["mate"] = int(score_token[1:])
            except ValueError:
                pass
        else:
            try:
                evaluation["cp"] = int(round(float(score_token) * 100))
            except ValueError:
                pass

    if depth_token:
        depth_token = depth_token.strip()
        if depth_token.isdigit():
            evaluation["depth"] = int(depth_token)

    for token in tokens[1:]:
        time_match = TIME_RE.search(token)
        if time_match:
            try:
                evaluation["time_seconds"] = float(time_match.group(1))
            except ValueError:
                pass

    return evaluation


def iter_game_samples(game: chess.pgn.Game, max_positions: Optional[int] = None) -> Iterator[dict]:
    """Yield one training sample per move in the provided PGN game."""
    board = game.board()
    result = game.headers.get("Result", "*")
    ply_count = 0

    node = game
    while node.variations:
        if max_positions is not None and ply_count >= max_positions:
            break

        next_node = node.variation(0)
        move = next_node.move
        comment = next_node.comment
        evaluation = parse_comment(comment)

        sample = {
            "fen": board.fen(),
            "uci": move.uci(),
            "result": result,
            "ply": ply_count,
            "evaluation": evaluation,
        }
        if comment:
            sample["comment"] = comment.strip()

        yield sample

        board.push(move)
        node = next_node
        ply_count += 1


def parse_pgn_file(path: Path, max_games: Optional[int], max_positions: Optional[int]) -> Iterator[dict]:
    """Stream parsed samples from a .pgn.gz archive."""
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        game_count = 0
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break

            yield from iter_game_samples(game, max_positions=max_positions)

            game_count += 1
            if max_games is not None and game_count >= max_games:
                break


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def download_subset(pattern: str, cache_dir: Path, repo_id: str) -> Path:
    """Download the requested subset of the dataset and return the local repo dir."""
    repo_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=pattern,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )
    return Path(repo_dir)


def generate_samples(
    pattern: str,
    cache_dir: Path,
    output: Path,
    repo_id: str = DEFAULT_REPO_ID,
    max_games: Optional[int] = None,
    max_positions: Optional[int] = None,
    position_cap: Optional[int] = 50000,
    overwrite: bool = False,
) -> int:
    ensure_output_path(output, overwrite)
    repo_dir = download_subset(pattern, cache_dir, repo_id=repo_id)
    pgn_files = sorted(repo_dir.rglob("*.pgn.gz"))
    if not pgn_files:
        raise FileNotFoundError(f"No PGN files found for pattern {pattern}")

    total_written = 0
    with output.open("w", encoding="utf-8") as out_file:
        for pgn_path in pgn_files:
            for sample in tqdm(
                parse_pgn_file(pgn_path, max_games, max_positions),
                desc=f"Parsing {pgn_path.name}",
            ):
                out_file.write(json.dumps(sample) + os.linesep)
                total_written += 1
                if position_cap and total_written >= position_cap:
                    print(f"Reached position cap ({position_cap}). Stopping.")
                    return total_written

    return total_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Fishtest PGNs into JSONL samples.")
    parser.add_argument("--pattern", type=str, default="24-12-*/*/*.pgn.gz", help="HF Hub allow_pattern glob.")
    parser.add_argument("--cache-dir", type=Path, default=Path("model_dataset/raw"), help="Where to cache downloads.")
    parser.add_argument("--output", type=Path, default=Path("model_dataset/fishtest_samples.jsonl"))
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--max-games", type=int, default=None, help="Limit number of games per file.")
    parser.add_argument("--max-positions", type=int, default=None, help="Limit number of moves per game.")
    parser.add_argument("--position-cap", type=int, default=50000, help="Stop after emitting this many samples.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists.")
    args = parser.parse_args()

    total = generate_samples(
        pattern=args.pattern,
        cache_dir=args.cache_dir,
        output=args.output,
        repo_id=args.repo_id,
        max_games=args.max_games,
        max_positions=args.max_positions,
        position_cap=args.position_cap,
        overwrite=args.overwrite,
    )
    print(f"Wrote {total} samples to {args.output}")


if __name__ == "__main__":
    main()
