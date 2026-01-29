import pandas as pd
import pytest

import segtraq as st


def test_nucleus_cell_similarity(sdata_new):
    corr_df = st.rc.nucleus_cell_similarity(sdata_new, n_jobs=8)
    assert isinstance(corr_df, pd.DataFrame), f"Expected DataFrame, got {type(corr_df)}"
    exp = {"cell_id", "nucleus_id", "IoU", "nucleus_fraction", "nucleus_cell_similarity"}
    assert set(corr_df.columns) == exp, f"Columns mismatch: expected {exp}, got {set(corr_df.columns)}"
    assert corr_df["nucleus_cell_similarity"].dtype == float, (
        f"Expected correlation dtype float, got {corr_df['nucleus_cell_similarity'].dtype}"
    )


def test_nucleus_cell_similarity_invalid_metric(sdata_new):
    with pytest.raises(ValueError, match="Metric dummy_metric not supported"):
        st.rc.nucleus_cell_similarity(sdata_new, metric="dummy_metric", n_jobs=8)
