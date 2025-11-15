"""
Quick evaluation script for the trained value network.

Usage:
    python test_model.py \
        --dataset model_dataset/fishtest_samples.jsonl \
        --model artifacts/value_net.pt \
        --limit 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from src.model import fen_to_planes, load_model
from src.move_encoding import move_to_index
import chess


def iter_samples(path: Path, limit: int | None) -> Iterator[dict]:
    """Stream JSONL samples with valid evaluations."""
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and count >= limit:
                break
            payload = json.loads(line)
            evaluation = payload.get("evaluation") or {}
            if "cp" not in evaluation:
                continue
            yield payload
            count += 1


def evaluate(model_path: Path, dataset_path: Path, limit: int | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(model_path, device=device)

    value_preds = []
    value_targets = []
    policy_matches = 0
    total_samples = 0

    for sample in iter_samples(dataset_path, limit):
        planes = fen_to_planes(sample["fen"])
        tensor = torch.from_numpy(planes).unsqueeze(0).to(device)

        board = chess.Board(sample["fen"])
        move = chess.Move.from_uci(sample["uci"])
        move_index = move_to_index(board, move)
        if move_index is None:
            continue

        with torch.no_grad():
            policy_logits, value_pred = model(tensor)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value_scalar = value_pred.cpu().numpy()[0]

        result = sample["result"]
        result_val = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}.get(result)
        if result_val is None:
            continue
        if board.turn == chess.BLACK:
            result_val = -result_val

        value_preds.append(float(value_scalar))
        value_targets.append(result_val)
        if np.argmax(policy) == move_index:
            policy_matches += 1
        total_samples += 1

    if total_samples == 0:
        raise RuntimeError("No valid samples found in dataset. Check the dataset path or generation step.")

    preds_arr = np.array(value_preds)
    targets_arr = np.array(value_targets)
    mse = float(np.mean((preds_arr - targets_arr) ** 2))
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    corr = float(np.corrcoef(preds_arr, targets_arr)[0, 1])
    policy_acc = policy_matches / total_samples

    print(f"Evaluated {total_samples} samples")
    print(f"Policy top-1 accuracy: {policy_acc:.4f}")
    print(f"Value MSE: {mse:.4f}")
    print(f"Value MAE: {mae:.4f}")
    print(f"Value correlation: {corr:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the chess value network on a JSONL dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to JSONL dataset.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/value_net.pt"), help="Path to trained model.")
    parser.add_argument("--limit", type=int, default=2000, help="Number of samples to evaluate (default: 2000).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.model, args.dataset, args.limit)


if __name__ == "__main__":
    main()
