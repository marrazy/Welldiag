from __future__ import annotations
 
import json
from dataclasses import dataclass
from pathlib import Path
 
import numpy as np
import pandas as pd
 
from welldiag.data.loading import TRANSIENT_OFFSET
from welldiag.data.preprocess import PROCESSED_DIR, DataConfig
 
PAD_LABEL = -1
 
 
@dataclass
class Normalizer:
    """Robust per-sensor scaling: (value - median) / spread, then clipped."""
 
    sensors: list[str]
    center: np.ndarray  # median of each sensor
    scale: np.ndarray  # IQR / 1.349, which equals the std for normally distributed data
    clip: float
 
    @classmethod
    def fit(cls, frames: list[pd.DataFrame], sensors: list[str], clip: float) -> Normalizer:
        """Learn medians and spreads from training frames. Masked readings are NaN and ignored."""
        values = pd.concat([f[sensors] for f in frames], ignore_index=True)
        center = values.median()
        scale = (values.quantile(0.75) - values.quantile(0.25)) / 1.349
        center = center.fillna(0.0).to_numpy(np.float32)
        scale = scale.where(scale > 0, 1.0).fillna(1.0).to_numpy(np.float32)
        return cls(sensors, center, scale, clip)
 
    def transform(self, values: np.ndarray) -> np.ndarray:
        z = (values - self.center) / self.scale
        z = np.clip(z, -self.clip, self.clip)
        return np.nan_to_num(z, nan=0.0).astype(np.float32)
 
    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "sensors": self.sensors,
                    "center": self.center.tolist(),
                    "scale": self.scale.tolist(),
                    "clip": self.clip,
                },
                indent=2,
            )
        )
 
    @classmethod
    def load(cls, path: Path) -> Normalizer:
        d = json.loads(path.read_text())
        return cls(
            d["sensors"],
            np.array(d["center"], np.float32),
            np.array(d["scale"], np.float32),
            d["clip"],
        )
 
 
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
    mask_cols = [f"mask_{s}" for s in sensors]
    pad = cfg.steps(cfg.window_minutes) - 1
    stride = cfg.steps(cfg.stride_minutes)
    min_hist = cfg.steps(cfg.min_history_minutes)
    n_feat = 2 * len(sensors)
 
    names, xs, ys, onsets, windows = [], [], [], [], []
    for i, (name, df) in enumerate(frames.items()):
        values = normalizer.transform(df[sensors].to_numpy(np.float32))
        masks = df[mask_cols].to_numpy(np.float32)
        label, onset = split_labels(df["class"].to_numpy())
 
        pad_x = np.zeros((pad, n_feat), np.float32)
        pad_x[:, len(sensors):] = 1.0  # padding is "no data" for every sensor
 
        names.append(name)
        xs.append(np.vstack([pad_x, np.hstack([values, masks])]))
        ys.append(np.concatenate([np.full(pad, PAD_LABEL, np.int64), label]))
        onsets.append(np.concatenate([np.zeros(pad, np.float32), onset]))
 
        first_end = pad + min_hist - 1
        last_end = pad + len(df) - 1
        ends = np.arange(first_end, last_end + 1, stride)
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
    normalizer = Normalizer.fit(list(train_frames.values()), cfg.sensors, cfg.clip)
    train = build_arrays(train_frames, normalizer, cfg)
    test = build_arrays(load_frames(test_names), normalizer, cfg)
    return train, test, normalizer
 