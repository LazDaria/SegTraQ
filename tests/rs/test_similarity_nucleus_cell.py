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


def test_similarity_nucleus_cell_invalid_select_by(sdata_new):
    with pytest.raises(ValueError, match="select_by must be 'iou' or 'nucleus_fraction'"):
        st.rs.similarity_nucleus_cell(
            sdata_new,
            select_by="dummy_metric",
            n_jobs=8,
        )

def test_similarity_nucleus_cell_value_range(sdata_new):
    df = st.rs.similarity_nucleus_cell(sdata_new, n_jobs=8)

    vals = df["similarity_nucleus_cell"].dropna()
    assert ((vals >= -1) & (vals <= 1)).all()

def test_similarity_nucleus_cell_high_thresholds_return_nan(sdata_new):
    df = st.rs.similarity_nucleus_cell(
        sdata_new,
        min_transcripts=10_000,
        min_genes=10_000,
        n_jobs=8,
    )

    assert df["similarity_nucleus_cell"].isna().all()

def test_similarity_nucleus_cell_no_inplace(sdata_new):
    before_cols = set(sdata_new.tables["table"].obs.columns)

    df = st.rs.similarity_nucleus_cell(
        sdata_new,
        n_jobs=8,
        inplace=False,
    )

    after_cols = set(sdata_new.tables["table"].obs.columns)

    assert isinstance(df, pd.DataFrame)
    assert before_cols == after_cols