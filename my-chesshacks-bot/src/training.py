from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F

try:  # Local package imports when running via `python -m`
    from .dataset import PolicyValueDataset
    from .model import create_model, save_model
except ImportError:  # Running as a script
    from dataset import PolicyValueDataset  # type: ignore
    from model import create_model, save_model  # type: ignore

log = logging.getLogger(__name__)


def _collate_batch(batch: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    planes, policy, value = zip(*batch)
    return torch.stack(planes), torch.stack(policy).long(), torch.stack(value).float()


def _batch_iterator(dataset: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], batch_size: int):
    batch = []
    for sample in dataset:
        batch.append(sample)
        if len(batch) == batch_size:
            yield _collate_batch(batch)
            batch = []
    if batch:
        yield _collate_batch(batch)


def train_supervised(
    data_paths: Sequence[Path | str],
    output_path: Path | str,
    epochs: int = 3,
    batch_size: int = 256,
    steps_per_epoch: int = 500,
    lr: float = 3e-4,
) -> Path:
    """
    Train the policy/value network on JSONL samples produced by download_model.py.
    """
    if not data_paths:
        raise ValueError("No dataset paths provided.")

    paths = [Path(p) for p in data_paths]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"The following dataset files are missing: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    value_loss_fn = torch.nn.MSELoss()

    samples_per_epoch = batch_size * steps_per_epoch

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_policy = 0.0
        epoch_value = 0.0
        steps = 0

        dataset = PolicyValueDataset(paths, max_samples=samples_per_epoch)
        for planes, policy_targets, value_targets in _batch_iterator(dataset, batch_size):
            planes = planes.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()
            policy_logits, value_pred = model(planes)
            policy_loss = F.cross_entropy(policy_logits, policy_targets)
            value_loss = value_loss_fn(value_pred.view(-1), value_targets)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_policy += policy_loss.item()
            epoch_value += value_loss.item()
            steps += 1
            if steps >= steps_per_epoch:
                break

        if steps < steps_per_epoch:
            raise RuntimeError("Dataset exhausted before reaching the requested steps_per_epoch.")

        avg_loss = epoch_loss / steps
        avg_policy = epoch_policy / steps
        avg_value = epoch_value / steps
        log.info(
            "Epoch %d/%d - loss: %.4f policy: %.4f value: %.4f",
            epoch,
            epochs,
            avg_loss,
            avg_policy,
            avg_value,
        )

    output = save_model(model, output_path)
    return output


# ---------------------------------------------------------------------------
# Legacy coach-based training scaffold (kept for backwards compatibility)
# ---------------------------------------------------------------------------
try:
    from selfTrainCoach import Coach
    from game import chess as Game
    from neuralTemplate import NNetWrapper as LegacyNN
    from tools import dotdict
except ImportError:  # pragma: no cover - only used when legacy stack available
    Coach = Game = LegacyNN = None

    class dotdict(dict):
        def __getattr__(self, item):
            return self[item]


legacy_args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})


def main():  # pragma: no cover - only for the legacy AlphaZero loop
    if Coach is None or Game is None or LegacyNN is None:
        raise RuntimeError("Legacy training components are unavailable in this environment.")

    log.info('Loading %s...', Game.__name__)
    g = Game(6)

    log.info('Loading %s...', LegacyNN.__name__)
    nnet = LegacyNN(g)

    if legacy_args.load_model:
        log.info('Loading checkpoint "%s/%s"...', legacy_args.load_folder_file[0], legacy_args.load_folder_file[1])
        nnet.load_checkpoint(legacy_args.load_folder_file[0], legacy_args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, legacy_args)

    if legacy_args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":  # pragma: no cover
    main()
