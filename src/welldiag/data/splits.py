from __future__ import annotations
 
from pathlib import Path
 
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
 
from welldiag.data.loading import PROJECT_ROOT
 
SYNTHETIC_FOLD = -1
FOLDS_PATH = PROJECT_ROOT / "configs" / "folds.csv"  # small, committed: everyone gets the same splits
 
 
def make_folds(instances: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Return a copy of `instances` with a `fold` column (0..n_splits-1, or -1 for synthetic)."""
    out = instances.copy()
    out["fold"] = SYNTHETIC_FOLD
 
    real = out[out["source"] == "real"]
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test_idx) in enumerate(cv.split(real, y=real["label"], groups=real["well"])):
        out.loc[real.index[test_idx], "fold"] = fold
    return out
 
 
def check_folds(folded: pd.DataFrame) -> list[str]:
    """Problems with a fold assignment; an empty list means the folds are usable."""
    problems = []
    real = folded[folded["source"] == "real"]
 
    # 1. No well may be split across folds
    split_wells = real.groupby("well")["fold"].nunique()
    for well in split_wells[split_wells > 1].index:
        problems.append(f"{well} appears in more than one fold")
 
    # 2. Every fault tested in a fold must also have real training data from other wells
    for fold in sorted(real["fold"].unique()):
        test = real[real["fold"] == fold]
        train = real[real["fold"] != fold]
        for label in sorted(test["label"].unique()):
            if label not in set(train["label"]):
                problems.append(f"fold {fold}: label {label} has no real training wells")
    return problems
 
 
def find_valid_folds(
    instances: pd.DataFrame, n_splits: int = 5, max_seeds: int = 200
) -> tuple[pd.DataFrame, int]:
    """Try seeds until the folds pass `check_folds`; return the folds and the seed used."""
    for seed in range(max_seeds):
        folded = make_folds(instances, n_splits=n_splits, seed=seed)
        if not check_folds(folded):
            return folded, seed
    raise RuntimeError(f"No valid {n_splits}-fold split found in {max_seeds} seeds")
 
 
def save_folds(folded: pd.DataFrame, path: Path) -> None:
    """Save instance -> fold mapping. Small enough to commit, so splits are reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    folded[["instance", "label", "source", "well", "fold"]].to_csv(path, index=False)
 
 
def load_folds(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
 
 
def fold_summary(folded: pd.DataFrame) -> pd.DataFrame:
    """Real instances per label in each test fold; use it to eyeball balance."""
    real = folded[folded["source"] == "real"]
    return pd.crosstab(real["label"], real["fold"], margins=True)
 
 
if __name__ == "__main__":
    from welldiag.data.preprocess import PROCESSED_DIR, load_config
 
    cfg = load_config()
    index = pd.read_csv(PROCESSED_DIR / "index.csv")
    folded, seed = find_valid_folds(index, n_splits=cfg.n_folds)
    save_folds(folded, FOLDS_PATH)
 
    print(f"Valid {cfg.n_folds}-fold split found with seed {seed}; saved to {FOLDS_PATH}")
    print("\nReal instances per label (rows) in each test fold (columns):")
    print(fold_summary(folded))