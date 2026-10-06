"""Load instances from the Petrobras 3W Dataset (v2, Parquet).

Each instance is one Parquet file in data/3W/dataset/<label>/.
The file name tells us the source:
    WELL-00001_20170201010207.parquet -> real, from well WELL-00001
    SIMULATED_00001.parquet           -> simulated
    DRAWN_00001.parquet               -> hand-drawn
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# src/welldiag/data/loading.py -> project root is three levels above this folder
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "3W" / "dataset"

TRANSIENT_OFFSET = 100  # label + 100 = transient phase of that event

EVENT_NAMES = {
    0: "Normal operation",
    1: "Abrupt increase of BSW",
    2: "Spurious closure of DHSV",
    3: "Severe slugging",
    4: "Flow instability",
    5: "Rapid productivity loss",
    6: "Quick restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in production line",
    9: "Hydrate in service line",
}

LABEL_COLUMNS = ["class", "state"]


def parse_source(stem: str) -> tuple[str, str | None]:
    """Return (source, well_id) from a file name without extension."""
    if stem.startswith("WELL-"):
        return "real", stem.split("_")[0]
    if stem.startswith("SIMULATED"):
        return "simulated", None
    if stem.startswith("DRAWN"):
        return "drawn", None
    return "unknown", None


def list_instances(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per instance: unique ID ("<label>_<file name>"), label, event, source, well, path."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"3W dataset not found at {data_dir}. "
            "Clone it with: git clone --depth 1 https://github.com/petrobras/3W.git data/3W"
        )

    rows = []
    for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        label = int(label_dir.name)
        for path in sorted(label_dir.glob("*.parquet")):
            source, well = parse_source(path.stem)
            rows.append(
                {
                    # File names repeat across folders (every folder has SIMULATED_00001),
                    # so the folder label is part of the ID
                    "instance": f"{label}_{path.stem}",
                    "label": label,
                    "event": EVENT_NAMES.get(label, "Unknown"),
                    "source": source,
                    "well": well,
                    "path": path,
                }
            )
    instances = pd.DataFrame(rows)
    duplicated = instances["instance"][instances["instance"].duplicated()]
    if not duplicated.empty:
        raise ValueError(f"Instance IDs must be unique; repeated: {duplicated.tolist()[:5]}")
    return instances


def load_instance(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read one instance. Index is the timestamp; pass `columns` to read only some."""
    df = pd.read_parquet(path, columns=columns)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """All columns except the label columns."""
    return [c for c in df.columns if c not in LABEL_COLUMNS]


def transient_mask(df: pd.DataFrame) -> pd.Series:
    """True where the row is in an event's transient phase (class >= 100)."""
    return df["class"].fillna(-1).astype(int) >= TRANSIENT_OFFSET