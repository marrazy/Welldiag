"""Step 4 of the data pipeline: PyTorch Dataset, class-balanced sampler, DataLoaders.

    train_loader, test_loader, normalizer = make_loaders(test_fold=0, cfg=load_config())
    for x, y_class, y_onset in train_loader:
        # x:       (batch, 2 * n_sensors, window_minutes) float32: values, then masks
        # y_class: (batch,) int64, event label 0-9 at the end of the window
        # y_onset: (batch,) float32, 1 if a fault is developing at the end of the window
        ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from welldiag.data.preprocess import DataConfig
from welldiag.data.splits import FOLDS_PATH, load_folds
from welldiag.data.windows import FoldArrays, prepare_fold, prepare_window


class WindowDataset(Dataset):
    """Serves 60-minute windows from FoldArrays, one at a time."""

    def __init__(self, arrays: FoldArrays, cfg: DataConfig, augment: bool = False, seed: int = 0):
        self.arrays = arrays
        self.cfg = cfg
        self.augment = augment
        self.n_sensors = len(cfg.sensors)
        self.length = cfg.steps(cfg.window_minutes)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.arrays)

    def __getitem__(self, k: int):
        i, end = self.arrays.windows[k]
        block = self.arrays.x[i][end - self.length + 1 : end + 1]
        z = block[:, : self.n_sensors].copy()
        mask = block[:, self.n_sensors :].copy()

        if self.augment and self.cfg.sensor_dropout > 0:
            # Blank random sensors for the whole window, whatever the class,
            # so "this sensor is missing" stops being a clue to the fault
            drop = self.rng.random(self.n_sensors) < self.cfg.sensor_dropout
            z[:, drop] = np.nan
            mask[:, drop] = 1.0

        z = prepare_window(z, self.cfg.centering, self.cfg.clip, self.cfg.squash)
        x = np.concatenate([z, mask], axis=1).T  # (channels, time), the layout Conv1d expects
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.tensor(self.arrays.y_class[i][end], dtype=torch.long),
            torch.tensor(self.arrays.y_onset[i][end], dtype=torch.float32),
        )


def balanced_weights(arrays: FoldArrays) -> np.ndarray:
    """Sampling weight per window so every class, and every instance within a class, counts equally.

    Without this, normal operation and productivity loss (tens of thousands of windows) would
    fill almost every batch, and DHSV (about 1,700) would rarely be seen.
    """
    df = pd.DataFrame({"label": arrays.window_labels(), "instance": arrays.windows[:, 0]})
    windows_in_group = df.groupby(["label", "instance"])["label"].transform("size")
    instances_in_class = df.groupby("label")["instance"].transform("nunique")
    n_classes = df["label"].nunique()
    return (1.0 / (n_classes * instances_in_class * windows_in_group)).to_numpy()


def make_loaders(
    test_fold: int,
    cfg: DataConfig,
    batch_size: int = 256,
    samples_per_epoch: int | None = None,
    seed: int = 0,
):
    """Train and test DataLoaders for one cross-validation fold.

    Training batches are drawn with replacement using `balanced_weights`, with sensor dropout.
    The test loader goes through every test window once, in order, without augmentation.
    """
    train, test, normalizer = prepare_fold(load_folds(FOLDS_PATH), test_fold, cfg)
    train_ds = WindowDataset(train, cfg, augment=True, seed=seed)
    test_ds = WindowDataset(test, cfg, augment=False)

    sampler = WeightedRandomSampler(
        torch.tensor(balanced_weights(train), dtype=torch.double),
        num_samples=samples_per_epoch or len(train_ds),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, normalizer