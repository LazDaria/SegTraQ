import pandas as pd
import pytest

import segtraq as st


def test_match_nuclei_to_cells(sdata_new):
    df = st.rs.match_nuclei_to_cells(sdata_new, n_jobs=8)
    assert isinstance(df, pd.DataFrame), f"cell_nucleus_match should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "nucleus_id", "iou", "nucleus_fraction"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"
    assert "iou" in sdata_new.tables["table"].obs.columns, "iou column not found in sdata_new.tables['table'].obs"
    cell_ids = df["cell_id"]
    assert cell_ids.is_unique, "Duplicate cell IDs found in cell_id column"
    nuc_ids = df["nucleus_id"].dropna()
    assert nuc_ids.is_unique, "Duplicate nucleus IDs found in nucleus_id column"


def test_match_nuclei_to_cells_invalid_select_by(sdata_new):
    with pytest.raises(ValueError, match="select_by must be"):
        st.rs.match_nuclei_to_cells(sdata_new, select_by="invalid_option", n_jobs=8, inplace=False)
