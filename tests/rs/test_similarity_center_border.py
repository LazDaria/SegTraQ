import pandas as pd

import segtraq as st


def test_similarity_center_border(sdata_new):
    df = st.rs.similarity_center_border(sdata_new)

    assert isinstance(df, pd.DataFrame), (
        f"similarity_center_border should return a DataFrame, got {type(df)}"
    )

    expected_cols = {"cell_id", "similarity_center_border"}
    assert set(df.columns) == expected_cols, (
        f"Expected columns {expected_cols}, but got {set(df.columns)}"
    )

    assert df["similarity_center_border"].dtype == float

def test_similarity_center_border_value_range(sdata_new):
    df = st.rs.similarity_center_border(sdata_new)

    vals = df["similarity_center_border"].dropna()
    assert ((vals >= -1) & (vals <= 1)).all()

def test_similarity_center_border_high_thresholds_return_nan(sdata_new):
    df = st.rs.similarity_center_border(
        sdata_new,
        min_transcripts=10_000,
        min_genes=10_000,
    )

    assert df["similarity_center_border"].isna().all()

def test_similarity_center_border_no_inplace(sdata_new):
    before_cols = set(sdata_new.tables["table"].obs.columns)

    df = st.rs.similarity_center_border(
        sdata_new,
        inplace=False,
    )

    after_cols = set(sdata_new.tables["table"].obs.columns)

    assert isinstance(df, pd.DataFrame)
    assert before_cols == after_cols