"""Step 3 of the data pipeline: normalise sensors and index 60-minute windows.

For one cross-validation fold:
1. `Normalizer.fit` learns each sensor's level and typical spread from the
   TRAINING instances only. Using test data here would leak information.
2. `build_arrays` scales every instance, pads its start, and lists every window.
3. `prepare_window` (called by the Dataset for each window) centres the window,
   compresses large values, clips glitches and replaces missing readings with 0.

Each instance becomes an array with one row per minute and 2 x n_sensors columns:
    [scaled sensor values ..., masks ...]
Scaled values keep NaN where a reading is missing; `prepare_window` turns them into 0.

The first `window_minutes - 1` rows are padding (no data, all masks 1), so a window
can end early in a recording. This keeps short instances usable and matches a live
system that has only just started collecting data.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from welldiag.data.loading import TRANSIENT_OFFSET
from welldiag.data.preprocess import PROCESSED_DIR, DataConfig

PAD_LABEL = -1


@dataclass
class Normalizer:
    """Per-sensor scaling: (value - center) / scale."""

    sensors: list[str]
    center: np.ndarray  # median of each sensor over all training data
    scale: np.ndarray  # typical spread of each sensor WITHIN one instance

    @classmethod
    def fit(cls, frames: list[pd.DataFrame], sensors: list[str]) -> Normalizer:
        """Learn centres and spreads from training frames. Masked readings are NaN and ignored.

        The spread is measured inside each instance, then the median across instances is used.
        Pooling all instances together would mostly measure the differences between wells
        (one well runs at 80 bar, another at 150), which would squash real fault changes.
        """
        center = pd.concat([f[sensors] for f in frames], ignore_index=True).median()
        spreads = pd.DataFrame(
            [(f[sensors].quantile(0.75) - f[sensors].quantile(0.25)) / 1.349 for f in frames]
        )  # IQR / 1.349 equals the std for normally distributed data
        scale = spreads.where(spreads > 0).median()
        return cls(
            sensors,
            center.fillna(0.0).to_numpy(np.float32),
            scale.fillna(1.0).to_numpy(np.float32),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Scale values; NaN (missing) stays NaN."""
        return ((values - self.center) / self.scale).astype(np.float32)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"sensors": self.sensors, "center": self.center.tolist(), "scale": self.scale.tolist()},
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> Normalizer:
        d = json.loads(path.read_text())
        return cls(d["sensors"], np.array(d["center"], np.float32), np.array(d["scale"], np.float32))


def prepare_window(z: np.ndarray, centering: str, clip: float, squash: str = "asinh") -> np.ndarray:
    """Final processing of one window of scaled values, shape (minutes, sensors).

    centering="window": subtract each sensor's median within this window, so the model sees
    how readings changed during the hour rather than which well it is on.
    centering="global": keep the training-wide centring from the Normalizer.

    squash="asinh": compress large values smoothly. asinh(x) is close to x for small x and
    grows like log(x) for large x, so a change of 100 spreads (5.3) still looks different from
    one of 10 (3.0) instead of both being cut off at the clip.

    Then clip extremes and set missing readings to 0 (their mask says they are missing).
    """
    if centering == "window":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-missing sensor: median is NaN
            level = np.nanmedian(z, axis=0)
        z = z - np.nan_to_num(level)
    elif centering != "global":
        raise ValueError(f"centering must be 'window' or 'global', got {centering!r}")

    if squash == "asinh":
        z = np.arcsinh(z)
    elif squash != "none":
        raise ValueError(f"squash must be 'asinh' or 'none', got {squash!r}")

    return np.nan_to_num(np.clip(z, -clip, clip), nan=0.0).astype(np.float32)


@dataclass
class FoldArrays:
    """Everything the PyTorch Dataset needs for one set of instances (train or test)."""

    names: list[str]
    x: list[np.ndarray]  # per instance: (rows, 2 * n_sensors) float32, padding included
    y_class: list[np.ndarray]  # per instance: (rows,) int64 event label 0-9, -1 in padding
    y_onset: list[np.ndarray]  # per instance: (rows,) float32, 1 during a fault's transient
    windows: np.ndarray  # (n_windows, 2) int64: [instance number, row where the window ends]

    def __len__(self) -> int:
        return len(self.windows)

    def window_labels(self) -> np.ndarray:
        """Event label (0-9) at the end of every window."""
        return np.array([self.y_class[i][e] for i, e in self.windows], dtype=np.int64)


def load_frames(names: list[str], processed_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    return {n: pd.read_parquet(processed_dir / f"{n}.parquet") for n in names}


def split_labels(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw 3W class (e.g. 108) -> (event label 8, onset flag 1)."""
    raw = raw.astype(np.int64)
    return raw % TRANSIENT_OFFSET, (raw >= TRANSIENT_OFFSET).astype(np.float32)


def build_arrays(
    frames: dict[str, pd.DataFrame], normalizer: Normalizer, cfg: DataConfig
) -> FoldArrays:
    sensors = normalizer.sensors
    n_sensors = len(sensors)
    mask_cols = [f"mask_{s}" for s in sensors]
    pad = cfg.steps(cfg.window_minutes) - 1
    stride = cfg.steps(cfg.stride_minutes)
    fault_stride = cfg.steps(cfg.fault_stride_minutes)
    min_hist = cfg.steps(cfg.min_history_minutes)

    names, xs, ys, onsets, windows = [], [], [], [], []
    for i, (name, df) in enumerate(frames.items()):
        values = normalizer.transform(df[sensors].to_numpy(np.float32))
        masks = df[mask_cols].to_numpy(np.float32)
        label, onset = split_labels(df["class"].to_numpy())

        pad_x = np.full((pad, 2 * n_sensors), np.nan, np.float32)
        pad_x[:, n_sensors:] = 1.0  # padding is "no data" for every sensor

        y = np.concatenate([np.full(pad, PAD_LABEL, np.int64), label])
        names.append(name)
        xs.append(np.vstack([pad_x, np.hstack([values, masks])]))
        ys.append(y)
        onsets.append(np.concatenate([np.zeros(pad, np.float32), onset]))

        # A window ends every `stride` minutes, and every `fault_stride` minutes during faults
        first_end = pad + min_hist - 1
        ends = np.arange(first_end, pad + len(df))
        offset = ends - first_end
        keep = (offset % stride == 0) | ((y[ends] != 0) & (offset % fault_stride == 0))
        ends = ends[keep]
        windows.append(np.column_stack([np.full(len(ends), i), ends]))

    windows_arr = np.vstack(windows) if windows else np.empty((0, 2), np.int64)
    return FoldArrays(names, xs, ys, onsets, windows_arr.astype(np.int64))


def fold_instances(folds: pd.DataFrame, test_fold: int) -> tuple[list[str], list[str]]:
    """(training instance names, test instance names) for one fold.

    Training = real instances from the other folds + all simulated instances (fold -1).
    """
    train = folds.loc[folds["fold"] != test_fold, "instance"].tolist()
    test = folds.loc[folds["fold"] == test_fold, "instance"].tolist()
    return train, test


def prepare_fold(folds: pd.DataFrame, test_fold: int, cfg: DataConfig):
    """Load, normalise and window one fold. Returns (train arrays, test arrays, normalizer)."""
    train_names, test_names = fold_instances(folds, test_fold)
    train_frames = load_frames(train_names)
    normalizer = Normalizer.fit(list(train_frames.values()), cfg.sensors)
    train = build_arrays(train_frames, normalizer, cfg)
    test = build_arrays(load_frames(test_names), normalizer, cfg)
    return train, test, normalizer