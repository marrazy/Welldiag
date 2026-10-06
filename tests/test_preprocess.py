import numpy as np
import pandas as pd
import pytest

from welldiag.data.preprocess import DataConfig, preprocess_instance, stuck_mask

CFG = DataConfig(
    sensors=["P-A", "P-B", "P-C", "P-ABSENT"],
    sources=["real"],
    resample="60s",
    max_fill_minutes=10,
    stuck_minutes=30,
    window_minutes=60,
    stride_minutes=5,
    min_history_minutes=15,
    fault_stride_minutes=1,
    centering="window",
    squash="asinh",
    clip=10,
    sensor_dropout=0.1,
    n_folds=5,
)


@pytest.fixture
def raw():
    """3 hours of fake 1 Hz data with known problems."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=3 * 3600, freq="s")
    df = pd.DataFrame({c: rng.normal(100, 1, len(idx)) for c in ["P-A", "P-B", "P-C"]}, index=idx)
    df.loc[idx[1800], "P-A"] = 1e9                           # one-second glitch spike
    df.loc[idx[3600:3600 + 40 * 60], "P-B"] = 42.0           # stuck for 40 minutes
    df.loc[idx[7200:7200 + 5 * 60], "P-C"] = np.nan          # 5-minute gap (should be filled)
    df.loc[idx[9000:9000 + 20 * 60], "P-C"] = np.nan         # 20-minute gap (should stay missing)
    df["class"] = pd.array([0] * 6000 + [108] * 3000 + [8] * 1800, dtype="Int64")
    return df


def test_shape_and_columns(raw):
    out = preprocess_instance(raw, CFG)
    assert len(out) == 180  # 3 hours -> 180 minutes
    assert list(out.columns) == CFG.sensors + [f"mask_{s}" for s in CFG.sensors] + ["class"]


def test_spike_removed_by_median(raw):
    out = preprocess_instance(raw, CFG)
    assert out["P-A"].max() < 110


def test_stuck_sensor_masked(raw):
    out = preprocess_instance(raw, CFG)
    stuck = out.loc["2024-01-01 01:00":"2024-01-01 01:39", "mask_P-B"]
    assert stuck.eq(1).all()
    assert out["mask_P-B"].sum() == 40  # exactly the stuck stretch, nothing else


def test_short_gap_filled_long_gap_masked(raw):
    out = preprocess_instance(raw, CFG)
    assert out.loc["2024-01-01 02:00":"2024-01-01 02:04", "mask_P-C"].eq(0).all()
    assert out.loc["2024-01-01 02:41":"2024-01-01 02:49", "mask_P-C"].eq(1).all()


def test_absent_sensor_fully_masked(raw):
    out = preprocess_instance(raw, CFG)
    assert out["mask_P-ABSENT"].eq(1).all()


def test_labels_kept(raw):
    out = preprocess_instance(raw, CFG)
    assert set(out["class"].unique()) == {0, 108, 8}


def test_stuck_mask_ignores_short_repeats():
    s = pd.DataFrame({"x": [1.0, 1.0, 2.0, 3.0, 3.0, 3.0]})
    assert stuck_mask(s, min_rows=3)["x"].tolist() == [False, False, False, True, True, True]