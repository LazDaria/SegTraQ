import pandas as pd

import segtraq as st
from segtraq.utils import _filter_control_and_low_quality_transcripts


def test_similarity_nucleus_cell(sdata_new):
    sdata = _filter_control_and_low_quality_transcripts(sdata_new)
    df = st.rs.similarity_nucleus_cell(sdata)

    assert isinstance(df, pd.DataFrame)

    exp = {
        "cell_id",
        "nucleus_id",
        "iou",
        "nucleus_fraction",
        "similarity_nucleus_cell",
        "similarity_nucleus_cell_p_value",
    }

    assert set(df.columns) == exp

    assert pd.api.types.is_float_dtype(df["similarity_nucleus_cell"])

    assert pd.api.types.is_float_dtype(df["similarity_nucleus_cell_p_value"])


def test_similarity_nucleus_cell_value_range(sdata_new):
    sdata = _filter_control_and_low_quality_transcripts(sdata_new)
    df = st.rs.similarity_nucleus_cell(sdata)

    vals = df["similarity_nucleus_cell"].dropna()

    assert ((vals >= -2) & (vals <= 2)).all()

    pvals = df["similarity_nucleus_cell_p_value"].dropna()

    assert ((pvals >= 0) & (pvals <= 1)).all()


def test_similarity_nucleus_cell_high_thresholds_return_nan(sdata_new):
    sdata = _filter_control_and_low_quality_transcripts(sdata_new)
    df = st.rs.similarity_nucleus_cell(
        sdata,
        min_transcripts=10_000,
        min_genes=10_000,
        n_jobs=8,
    )

    assert df["similarity_nucleus_cell"].isna().all()
    assert df["similarity_nucleus_cell_p_value"].isna().all()


def test_similarity_nucleus_cell_no_inplace(sdata_new):
    sdata = _filter_control_and_low_quality_transcripts(sdata_new)
    before_cols = set(sdata.tables["table"].obs.columns)

    df = st.rs.similarity_nucleus_cell(
        sdata,
        n_jobs=8,
        inplace=False,
    )

    after_cols = set(sdata.tables["table"].obs.columns)

    assert isinstance(df, pd.DataFrame)
    assert before_cols == after_cols
