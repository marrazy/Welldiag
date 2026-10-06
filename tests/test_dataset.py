import numpy as np
import pandas as pd
import torch

from welldiag.data.dataset import WindowDataset, balanced_weights
from welldiag.data.windows import Normalizer, build_arrays

from test_windows import CFG, frame


def arrays_with_classes():
    frames = {
        "normal": frame(600, seed=1),
        "fault_a": frame(300, labels=[0] * 100 + [106] * 200, seed=2),
        "fault_b": frame(300, labels=[0] * 250 + [102] * 50, seed=3),
    }
    norm = Normalizer.fit(list(frames.values()), CFG.sensors)
    return build_arrays(frames, norm, CFG)


def test_item_shapes_and_types():
    ds = WindowDataset(arrays_with_classes(), CFG)
    x, y, onset = ds[0]
    assert x.shape == (4, 60)  # (2 * sensors, minutes)
    assert x.dtype == torch.float32 and y.dtype == torch.long and onset.dtype == torch.float32
    assert not torch.isnan(x).any()


def test_padding_shows_as_masked_zeros():
    ds = WindowDataset(arrays_with_classes(), CFG)
    x, _, _ = ds[0]  # first window: 15 real minutes + 45 padded
    assert (x[2:, :45] == 1).all()  # masks
    assert (x[:2, :45] == 0).all()  # values


def test_sensor_dropout_masks_whole_sensor():
    cfg = CFG.__class__(**{**CFG.__dict__, "sensor_dropout": 1.0})
    ds = WindowDataset(arrays_with_classes(), cfg, augment=True)
    x, _, _ = ds[10]
    assert (x[2:] == 1).all() and (x[:2] == 0).all()


def test_balanced_weights_give_each_class_equal_share():
    arrays = arrays_with_classes()
    w = balanced_weights(arrays)
    share = pd.Series(w).groupby(arrays.window_labels()).sum()
    assert np.allclose(share.to_numpy(), 1 / len(share))