import pandas as pd

import segtraq as st


def test_compute_cell_nuc_ious_structure(sdata_new):
    df = st.nc.compute_cell_nuc_ious(sdata_new, use_progress=False)

    assert isinstance(df, pd.DataFrame), f"compute_cell_nuc_ious should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "best_nuc_id", "IoU"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"

def test_compute_cell_nuc_ious_structure(sdata_new):
    assert False, sdata_new['table'].var_names

