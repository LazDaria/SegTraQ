import pandas as pd
import pytest

import segtraq as st


def test_similarity_nucleus_cell(sdata_new):
    corr_df = st.rs.similarity_nucleus_cell(sdata_new, n_jobs=8)
    assert isinstance(corr_df, pd.DataFrame), f"Expected DataFrame, got {type(corr_df)}"
    exp = {"cell_id", "nucleus_id", "iou", "nucleus_fraction", "similarity_nucleus_cell"}
    assert set(corr_df.columns) == exp, f"Columns mismatch: expected {exp}, got {set(corr_df.columns)}"
    assert corr_df["similarity_nucleus_cell"].dtype == float, (
        f"Expected correlation dtype float, got {corr_df['similarity_nucleus_cell'].dtype}"
    )


def test_similarity_nucleus_cell_invalid_metric(sdata_new):
    with pytest.raises(ValueError, match="Metric dummy_metric not supported"):
        st.rs.similarity_nucleus_cell(sdata_new, metric="dummy_metric", n_jobs=8)
