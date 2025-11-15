## High-Elo Training Roadmap

Goal: produce a ~2300-strength bot without exceeding \$26 of Modal credits.

### 1. Bulk Dataset Generation (Modal CPU)

1. Mirror or select a stronger dataset (examples):
   - `official-stockfish/fishtest_pgns` (current default) but with `position_cap` in the 1–5M range.
   - Lichess Elite/masters: upload the PGN dump to your own HF dataset repo (e.g. `ayan/chess_elite`) and pass `--repo-id ayan/chess_elite`.
   - LCZero training chunks or self-play logs (convert to PGN/JSONL first).
2. From the repo root (with Modal CLI logged in):
   ```bash
   modal run modal_app.py::build_dataset \
     --pattern "24-12-*/*/*.pgn.gz" \
     --position-cap 2000000 \
     --output-path /model_storage/datasets/fishtest_2m.jsonl \
     --max-games 2000
   ```
3. Credits impact: CPU + storage only (~\$1–2 per million positions).

### 2. Training Runs (Modal GPU)

Target: 40–60 GPU hours on T4 or A10G (fits \$26 budget).

Example command:
```bash
modal run modal_app.py::train_model \
  --dataset-path /model_storage/datasets/fishtest_2m.jsonl \
  --epochs 30 \
  --steps-per-epoch 1500 \
  --batch-size 512 \
  --lr 2e-4 \
  --output-name value_net_large.pt
```

Tips:
- Increase `trunk_channels` / residual blocks in `src/model.py` before training.
- Use LR decay: split training into multiple runs and resume from checkpoints.
- After each run:
  ```bash
  modal volume get chess-models checkpoints/value_net_large.pt artifacts/value_net.pt --force
  python test_model.py --dataset model_dataset/fishtest_2m.jsonl --model artifacts/value_net.pt --limit 5000
  ```

### 3. Self-Play Augmentation (Optional)

Use Modal CPU pools to run `selfTrainCoach` or a simplified MCTS loop to create self-play PGNs. Append them to HF dataset and rerun `download_model.py` with `--repo-id your/selfplay_repo`.

### 4. Inference Search Upgrades (Local, no credits)

- Increase `SEARCH_DEPTH` in `src/main.py`, add transposition tables, iterative deepening.
- Optionally integrate `src/mcts.py` for AlphaZero-style rollouts guided by the trained network.

### Budget Summary

| Item | Credits |
|------|---------|
| Dataset downloads + storage | \$3–5 |
| GPU training (T4/A10G) | \$18–22 |
| Self-play CPU jobs | \$0–3 |

Total: ≤ \$26 with careful monitoring (`modal balance`, `modal token limits`). Always run shorter training jobs first to validate hyperparameters before launching multi-hour sessions.
