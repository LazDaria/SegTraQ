import pandas as pd

import segtraq as st


def test_similarity_nucleus_cytoplasm(sdata_new):
    df = st.rs.similarity_nucleus_cytoplasm(sdata_new)

    assert isinstance(df, pd.DataFrame), f"similarity_nucleus_cytoplasm should return a DataFrame, got {type(df)}"

    expected_cols = {
        "cell_id",
        "nucleus_id",
        "iou",
        "nucleus_fraction",
        "similarity_nucleus_cytoplasm",
        "similarity_nucleus_cytoplasm_p_value",
    }

    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"

    # Cells without a matched nucleus should not have a similarity or p-value.
    mask = df["nucleus_id"].isna()

    assert df.loc[mask, "similarity_nucleus_cytoplasm"].isna().all()

    assert df.loc[mask, "similarity_nucleus_cytoplasm_p_value"].isna().all()

    # Output types
    assert pd.api.types.is_float_dtype(df["similarity_nucleus_cytoplasm"])

    assert pd.api.types.is_float_dtype(df["similarity_nucleus_cytoplasm_p_value"])


def test_similarity_nucleus_cytoplasm_value_range(sdata_new):
    df = st.rs.similarity_nucleus_cytoplasm(sdata_new)

    # Residual = observed cosine - mean null cosine
    vals = df["similarity_nucleus_cytoplasm"].dropna()
    assert ((vals >= -2) & (vals <= 2)).all()

    pvals = df["similarity_nucleus_cytoplasm_p_value"].dropna()
    assert ((pvals >= 0) & (pvals <= 1)).all()


def test_similarity_nucleus_cytoplasm_high_thresholds_return_nan(sdata_new):
    df = st.rs.similarity_nucleus_cytoplasm(
        sdata_new,
        min_transcripts=10_000,
        min_genes=10_000,
    )

    assert df["similarity_nucleus_cytoplasm"].isna().all()
    assert df["similarity_nucleus_cytoplasm_p_value"].isna().all()


def test_similarity_nucleus_cytoplasm_no_inplace(sdata_new):
    before_cols = set(sdata_new.tables["table"].obs.columns)

    df = st.rs.similarity_nucleus_cytoplasm(
        sdata_new,
        inplace=False,
    )

    after_cols = set(sdata_new.tables["table"].obs.columns)

    assert isinstance(df, pd.DataFrame)
    assert before_cols == after_cols
