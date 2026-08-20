import pandas as pd
import pytest

import segtraq as st

st.settings.n_jobs = -1


def test_match_nuclei_to_cells(sdata_new):
    df = st.rs.match_nuclei_to_cells(sdata_new, n_jobs=8)

    assert isinstance(df, pd.DataFrame), f"match_nuclei_to_cells should return a DataFrame, got {type(df)}"

    expected_cols = {"cell_id", "nucleus_id", "iou", "nucleus_fraction"}
    assert set(df.columns) == expected_cols

    assert {"nucleus_id", "iou", "nucleus_fraction"}.issubset(sdata_new.tables["table"].obs.columns)

    assert df["cell_id"].is_unique
    assert df["nucleus_id"].dropna().is_unique


def test_match_nuclei_to_cells_invalid_select_by(sdata_new):
    with pytest.raises(ValueError, match="select_by must be"):
        st.rs.match_nuclei_to_cells(sdata_new, select_by="invalid_option", n_jobs=8, inplace=False)
