import segtraq as st
import pytest
import pandas as pd

def test_compute_cell_nuc_ious_structure(sdata):
    df = st.nc.compute_cell_nuc_ious(sdata, use_progress=False)

    assert isinstance(df, pd.DataFrame), f"compute_cell_nuc_ious should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "best_nuc_id", "IoU"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"

    cell_ids = set(df['cell_id'])
    assert cell_ids == {100, 200}, f"Expected cell_ids {{100, 200}}, but got {cell_ids}"

    row = df.set_index('cell_id').loc[100]
    assert row['best_nuc_id'] == 10, f"For cell_id=100 expected best_nuc_id=10, got {row['best_nuc_id']}"
    assert row['IoU'] == pytest.approx(1.0), f"For cell_id=100 expected IoU≈1.0, got {row['IoU']}"
