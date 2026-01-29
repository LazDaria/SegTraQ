import pandas as pd

import segtraq as st


def test_compute_cell_nuc_match(sdata_new):
    df = st.rc.compute_cell_nuc_match(sdata_new, n_jobs=8)

    assert isinstance(df, pd.DataFrame), f"compute_cell_nuc_match should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "best_nuc_id", "IoU", "nucleus_fraction"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"
    assert "IoU" in sdata_new.tables["table"].obs.columns, "IoU column not found in sdata_new.tables['table'].obs"
    nuc_ids = df["best_nuc_id"].dropna()
    assert nuc_ids.is_unique, "Duplicate nucleus IDs found in best_nuc_id column"
