"""Step 1 of the data pipeline: clean and downsample each 3W instance.

For one instance, `preprocess_instance` returns one row per minute with:
    <sensor>       median reading in that minute (raw units), NaN where unusable
    mask_<sensor>  1.0 where the sensor had no usable value, else 0.0
    class          the event label at the end of that minute

Run `uv run python -m welldiag.data.preprocess` to process every instance once
and cache the results in data/processed/, so later steps never re-read raw files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from welldiag.data.loading import PROJECT_ROOT, list_instances, load_instance

CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class DataConfig:
    sensors: list[str]
    sources: list[str]
    resample: str
    max_fill_minutes: int
    stuck_minutes: int
    window_minutes: int
    stride_minutes: int
    min_history_minutes: int
    fault_stride_minutes: int
    centering: str
    squash: str
    clip: float
    sensor_dropout: float
    n_folds: int

    def steps(self, minutes: int) -> int:
        """Convert a duration in minutes to a number of rows after resampling."""
        step_minutes = pd.Timedelta(self.resample).total_seconds() / 60
        return max(1, round(minutes / step_minutes))


def load_config(path: Path = CONFIG_PATH) -> DataConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    with open(PROJECT_ROOT / raw.pop("sensors_file")) as f:
        sensors = yaml.safe_load(f)["sensors"]
    return DataConfig(sensors=sensors, **raw)


def stuck_mask(s: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    """True where a sensor repeats exactly the same value for at least `min_rows` rows in a row.

    A working sensor's per-minute median practically never repeats exactly,
    so long runs of identical values mean the sensor is frozen.
    """
    out = {}
    for col in s.columns:
        x = s[col]
        run_id = (x != x.shift()).cumsum()  # a new id every time the value changes
        run_len = x.groupby(run_id).transform("size")
        out[col] = (run_len >= min_rows) & x.notna()
    return pd.DataFrame(out, index=s.index)


def preprocess_instance(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Clean and downsample one raw instance (see module docstring for the output)."""
    # Sensors missing from this file become all-NaN columns, so every instance has the same shape
    sensors = df.reindex(columns=cfg.sensors)

    # 1. Downsample: the median of each minute also removes single-second glitch spikes
    s = sensors.resample(cfg.resample).median()
    labels = df["class"].resample(cfg.resample).last()  # label at the end of each minute

    # 2. Fill short gaps with the last good value; longer gaps stay missing
    s = s.ffill(limit=cfg.steps(cfg.max_fill_minutes))

    # 3. Treat stuck stretches as missing (after filling, so they are never filled back in)
    s = s.mask(stuck_mask(s, cfg.steps(cfg.stuck_minutes)))

    # 4. Mask: 1 where there is still no usable value
    mask = s.isna().astype("float32").add_prefix("mask_")

    out = pd.concat([s.astype("float32"), mask], axis=1)
    out["class"] = labels
    return out[out["class"].notna()]  # drop minutes with no recorded data at all


def build_processed(cfg: DataConfig, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Preprocess every instance from the configured sources and cache it to `out_dir`.

    Already-processed instances are skipped, so the script can be re-run safely.
    Delete `out_dir` after changing data.yaml or sensors.yaml.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = list_instances()
    instances = instances[instances["source"].isin(cfg.sources)].reset_index(drop=True)

    minutes = []
    for i, row in enumerate(instances.itertuples(), start=1):
        target = out_dir / f"{row.instance}.parquet"
        if target.exists():
            processed = pd.read_parquet(target, columns=["class"])
        else:
            processed = preprocess_instance(load_instance(row.path), cfg)
            processed.to_parquet(target)
        minutes.append(len(processed))
        if i % 100 == 0 or i == len(instances):
            print(f"{i}/{len(instances)} instances processed")

    index = instances.drop(columns="path").assign(minutes=minutes)
    index.to_csv(out_dir / "index.csv", index=False)
    return index


if __name__ == "__main__":
    build_processed(load_config())