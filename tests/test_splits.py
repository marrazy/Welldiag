import pandas as pd
 
from welldiag.data.splits import SYNTHETIC_FOLD, check_folds, find_valid_folds
 
 
def fake_index():
    """10 wells. Label 1 lives in only 2 wells, like quick restriction in the real data."""
    rows = []
    for w in range(10):
        well = f"WELL-{w:05d}"
        for k in range(5):
            rows.append({"instance": f"{well}_{k}", "label": 0, "source": "real", "well": well})
        if w < 5:
            rows.append({"instance": f"{well}_f2", "label": 2, "source": "real", "well": well})
        if w in (3, 7):
            rows.append({"instance": f"{well}_f1", "label": 1, "source": "real", "well": well})
    for k in range(20):
        rows.append({"instance": f"SIMULATED_{k}", "label": 1, "source": "simulated", "well": None})
    return pd.DataFrame(rows)
 
 
def test_valid_folds_pass_checks():
    folded, _ = find_valid_folds(fake_index(), n_splits=5)
    assert check_folds(folded) == []
 
 
def test_each_well_in_exactly_one_fold():
    folded, _ = find_valid_folds(fake_index(), n_splits=5)
    real = folded[folded["source"] == "real"]
    assert real.groupby("well")["fold"].nunique().eq(1).all()
 
 
def test_synthetic_always_train():
    folded, _ = find_valid_folds(fake_index(), n_splits=5)
    assert folded.loc[folded["source"] == "simulated", "fold"].eq(SYNTHETIC_FOLD).all()
 
 
def test_two_well_label_split_apart():
    folded, _ = find_valid_folds(fake_index(), n_splits=5)
    label1 = folded[(folded["label"] == 1) & (folded["source"] == "real")]
    assert label1["fold"].nunique() == 2  # the two wells must land in different folds
 
 
def test_check_catches_label_with_no_training_well():
    df = fake_index()
    df["fold"] = SYNTHETIC_FOLD
    real = df["source"] == "real"
    df.loc[real, "fold"] = 0
    df.loc[real & df["well"].isin(["WELL-00000", "WELL-00001"]), "fold"] = 1
    df.loc[real & df["well"].isin(["WELL-00003", "WELL-00007"]), "fold"] = 2  # both label-1 wells together
    assert any("label 1" in p for p in check_folds(df))