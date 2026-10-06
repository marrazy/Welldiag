import numpy as np
import pandas as pd
import pytest
 
from welldiag.data.preprocess import DataConfig
from welldiag.data.windows import PAD_LABEL, Normalizer, build_arrays, fold_instances, split_labels
 
CFG = DataConfig(
    sensors=["A", "B"],
    sources=["real"],
    resample="60s",
    max_fill_minutes=10,
    stuck_minutes=120,
    window_minutes=60,
    stride_minutes=5,
    min_history_minutes=15,
    clip=10,
    n_folds=5,
)
 
 
def frame(n_minutes, a=100.0, b=50.0, label=0):
    idx = pd.date_range("2024-01-01", periods=n_minutes, freq="min")
    rng = np.random.default_rng(n_minutes)
    df = pd.DataFrame(
        {"A": a + rng.normal(0, 1, n_minutes), "B": b + rng.normal(0, 2, n_minutes)}, index=idx
    )
    df["mask_A"] = 0.0
    df["mask_B"] = 0.0
    df["class"] = label
    return df
 
 
def test_normalizer_centres_and_scales():
    norm = Normalizer.fit([frame(500)], ["A", "B"], clip=10)
    z = norm.transform(frame(500)[["A", "B"]].to_numpy(np.float32))
    assert abs(np.median(z[:, 0])) < 0.1
    assert 0.7 < z[:, 0].std() < 1.3
 
 
def test_normalizer_ignores_masked_values_and_clips():
    f = frame(200)
    f.loc[f.index[:50], "A"] = np.nan  # masked readings
    norm = Normalizer.fit([f], ["A", "B"], clip=10)
    assert not np.isnan(norm.center).any()
    z = norm.transform(np.array([[np.nan, 1e9]], np.float32))
    assert z[0, 0] == 0.0  # masked -> 0
    assert z[0, 1] == 10.0  # clipped
 
 
def test_split_labels():
    label, onset = split_labels(np.array([0, 108, 8]))
    assert label.tolist() == [0, 8, 8]
    assert onset.tolist() == [0.0, 1.0, 0.0]
 
 
def test_padding_and_window_positions():
    norm = Normalizer.fit([frame(300)], ["A", "B"], clip=10)
    arrays = build_arrays({"x": frame(300)}, norm, CFG)
    x, y = arrays.x[0], arrays.y_class[0]
    assert x.shape == (59 + 300, 4)
    assert (x[:59, 2:] == 1).all() and (x[:59, :2] == 0).all()  # padding: no data, all masked
    assert (y[:59] == PAD_LABEL).all()
    ends = arrays.windows[:, 1]
    assert ends[0] == 59 + 15 - 1  # first window after 15 real minutes
    assert ends[-1] <= 59 + 300 - 1
    assert (np.diff(ends) == 5).all()
    assert (y[ends] != PAD_LABEL).all()  # every window ends on a real minute
 
 
def test_short_instance_still_produces_windows():
    norm = Normalizer.fit([frame(300)], ["A", "B"], clip=10)
    arrays = build_arrays({"short": frame(28)}, norm, CFG)
    assert len(arrays) > 0
 
 
def test_fold_instances_keeps_synthetic_in_training():
    folds = pd.DataFrame(
        {"instance": ["r0", "r1", "sim"], "fold": [0, 1, -1]},
    )
    train, test = fold_instances(folds, test_fold=0)
    assert set(train) == {"r1", "sim"} and test == ["r0"]
 