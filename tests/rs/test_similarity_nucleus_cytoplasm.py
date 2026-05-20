import numpy as np
import pandas as pd

import segtraq as st


def test_similarity_nucleus_cytoplasm(sdata_new):
    df = st.rs.similarity_nucleus_cytoplasm(sdata_new)
    # identify cells with missing nucleus_id
    mask = df["nucleus_id"].isna()

    # test that there is nothing computed for a cell without a nucleus
    for cell_id in df.loc[mask, "cell_id"]:
        corr = df.loc[df["cell_id"] == cell_id, "similarity_nucleus_cytoplasm"].iloc[0]
        assert np.isnan(corr), f"Expected NaN for cell {cell_id} with missing nucleus, got {corr}"

    # test that there is a valid correlation for cells with nucleus
    assert isinstance(df, pd.DataFrame), f"similarity_nucleus_cytoplasm should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "nucleus_id", "iou", "similarity_nucleus_cytoplasm", "nucleus_fraction"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"


def test_similarity_nucleus_cytoplasm_value_range(sdata_new):
    df = st.rs.similarity_nucleus_cytoplasm(sdata_new)

    vals = df["similarity_nucleus_cytoplasm"].dropna()
    assert ((vals >= -1) & (vals <= 1)).all()


def test_similarity_nucleus_cytoplasm_high_thresholds_return_nan(sdata_new):
    df = st.rs.similarity_nucleus_cytoplasm(
        sdata_new,
        min_transcripts=10_000,
        min_genes=10_000,
    )

    assert df["similarity_nucleus_cytoplasm"].isna().all()


def test_similarity_nucleus_cytoplasm_no_inplace(sdata_new):
    before_cols = set(sdata_new.tables["table"].obs.columns)

    df = st.rs.similarity_nucleus_cytoplasm(
        sdata_new,
        inplace=False,
    )

    after_cols = set(sdata_new.tables["table"].obs.columns)

    assert isinstance(df, pd.DataFrame)
    assert before_cols == after_cols


def test_similarity_nucleus_cytoplasm_different_table_and_shape_ids(sdata_new_table_ids):
    df = st.rs.similarity_nucleus_cytoplasm(
        sdata_new_table_ids,
        tables_cell_id_key="table_cell_id",
        points_cell_id_key="table_cell_id",
        inplace=False,
    )

    assert isinstance(df, pd.DataFrame)
    assert "table_cell_id" in df.columns
    assert not df["similarity_nucleus_cytoplasm"].isna().all()
