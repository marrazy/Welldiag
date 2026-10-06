import numpy as np
import pandas as pd

from welldiag.data.preprocess import DataConfig
from welldiag.data.windows import (
    PAD_LABEL,
    Normalizer,
    build_arrays,
    fold_instances,
    prepare_window,
    split_labels,
)

CFG = DataConfig(
    sensors=["A", "B"],
    sources=["real"],
    resample="60s",
    max_fill_minutes=10,
    stuck_minutes=120,
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


def frame(n_minutes, a=100.0, b=50.0, labels=None, seed=0):
    idx = pd.date_range("2024-01-01", periods=n_minutes, freq="min")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"A": a + rng.normal(0, 1, n_minutes), "B": b + rng.normal(0, 2, n_minutes)}, index=idx
    )
    df["mask_A"] = 0.0
    df["mask_B"] = 0.0
    df["class"] = 0 if labels is None else labels
    return df


def test_scale_is_within_instance_spread_not_between_wells():
    # Two "wells" at very different levels, each with std 1 on sensor A
    norm = Normalizer.fit([frame(500, a=100, seed=1), frame(500, a=1000, seed=2)], ["A", "B"])
    assert 0.8 < norm.scale[0] < 1.2  # pooled spread would be ~450


def test_transform_keeps_missing_as_nan():
    norm = Normalizer.fit([frame(200)], ["A", "B"])
    z = norm.transform(np.array([[np.nan, 50.0]], np.float32))
    assert np.isnan(z[0, 0]) and not np.isnan(z[0, 1])


def test_prepare_window_centres_clips_and_fills():
    z = np.array([[100.0, np.nan], [101.0, np.nan], [1e6, np.nan]], np.float32)
    out = prepare_window(z, centering="window", clip=10, squash="none")
    assert out[1, 0] == 0.0  # centred on the window median (101)
    assert out[2, 0] == 10.0  # clipped
    assert (out[:, 1] == 0).all()  # missing -> 0
    assert prepare_window(z, centering="global", clip=10, squash="none")[0, 0] == 10.0


def test_asinh_keeps_small_changes_and_separates_big_ones():
    z = np.array([[0.0], [0.5], [10.0], [100.0], [1e9]], np.float32)
    out = prepare_window(z, centering="global", clip=10, squash="asinh")[:, 0]
    assert abs(out[1] - 0.5) < 0.03  # small change barely altered
    assert out[2] < out[3] < 10  # 10x and 100x changes stay distinguishable, below the clip
    assert out[4] == 10.0  # only absurd glitches hit the clip


def test_split_labels():
    label, onset = split_labels(np.array([0, 108, 8]))
    assert label.tolist() == [0, 8, 8]
    assert onset.tolist() == [0.0, 1.0, 0.0]


def test_padding_and_normal_stride():
    norm = Normalizer.fit([frame(300)], ["A", "B"])
    arrays = build_arrays({"x": frame(300)}, norm, CFG)
    x, y = arrays.x[0], arrays.y_class[0]
    assert x.shape == (59 + 300, 4)
    assert np.isnan(x[:59, :2]).all() and (x[:59, 2:] == 1).all()  # padding: no data, all masked
    assert (y[:59] == PAD_LABEL).all()
    ends = arrays.windows[:, 1]
    assert ends[0] == 59 + 15 - 1  # first window after 15 real minutes
    assert (np.diff(ends) == 5).all()  # normal operation: every 5 minutes
    assert (y[ends] != PAD_LABEL).all()  # every window ends on a real minute


def test_fault_minutes_get_a_window_every_minute():
    labels = [0] * 200 + [106] * 12 + [6] * 10  # a 22-minute real quick restriction
    norm = Normalizer.fit([frame(222)], ["A", "B"])
    arrays = build_arrays({"x": frame(222, labels=labels)}, norm, CFG)
    fault_windows = (arrays.window_labels() == 6).sum()
    assert fault_windows == 22


def test_short_instance_still_produces_windows():
    norm = Normalizer.fit([frame(300)], ["A", "B"])
    arrays = build_arrays({"short": frame(28)}, norm, CFG)
    assert len(arrays) > 0


def test_fold_instances_keeps_synthetic_in_training():
    folds = pd.DataFrame({"instance": ["r0", "r1", "sim"], "fold": [0, 1, -1]})
    train, test = fold_instances(folds, test_fold=0)
    assert set(train) == {"r1", "sim"} and test == ["r0"]