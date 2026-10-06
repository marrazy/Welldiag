import pandas as pd
import pytest


from welldiag.data.loading import DATA_DIR, list_instances, parse_source, transient_mask


def test_parse_source_real():
    assert parse_source("WELL-00001_20170201010207") == ("real", "WELL-00001")


def test_parse_source_simulated_and_drawn():
    assert parse_source("SIMULATED_00001") == ("simulated", None)
    assert parse_source("DRAWN_00001") == ("drawn", None)


def test_transient_mask():
    df = pd.DataFrame({"class": pd.array([0, 108, 8, None], dtype="Int64")})
    assert transient_mask(df).tolist() == [False, True, False, False]


@pytest.mark.skipif(not DATA_DIR.exists(), reason="3W dataset not downloaded")
def test_list_instances_finds_all_labels():
    instances = list_instances()
    assert len(instances) > 0
    assert set(instances["label"]) <= set(range(10))
    assert "real" in set(instances["source"])


def test_instance_ids_unique_when_file_names_repeat(tmp_path):
    """Every 3W folder has SIMULATED_00001; IDs must still be unique."""
    idx = pd.date_range("2024-01-01", periods=3, freq="s")
    for label in (1, 6):
        (tmp_path / str(label)).mkdir()
        pd.DataFrame({"class": [0, 0, label]}, index=idx).to_parquet(
            tmp_path / str(label) / "SIMULATED_00001.parquet"
        )
    instances = list_instances(tmp_path)
    assert instances["instance"].is_unique
    assert set(instances["instance"]) == {"1_SIMULATED_00001", "6_SIMULATED_00001"}