import pandas as pd
import pytest

import segtraq as st


def test_border_admixture_score_type(sdata_new):
    df = st.rs.border_admixture_score(sdata_new)

    assert isinstance(df, pd.DataFrame), f"border_admixture_score should return a DataFrame, got {type(df)}"

    expected_cols = {
        "cell_id",
        "border_admixture_score",
        "border_admixture_p_value",
    }

    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"

    assert pd.api.types.is_float_dtype(df["border_admixture_score"])
    assert pd.api.types.is_float_dtype(df["border_admixture_p_value"])


def test_border_admixture_score_values(sdata_new):
    df = st.rs.border_admixture_score(sdata_new)

    # Residual = observed admixture score - mean null admixture score.
    # Both observed and null scores are in [0, 1], so the residual is in [-1, 1].
    vals = df["border_admixture_score"].dropna()
    assert vals.between(-1, 1).all()

    pvals = df["border_admixture_p_value"].dropna()
    assert pvals.between(0, 1).all()


def test_border_admixture_score_high_thresholds_return_nan(sdata_new):
    df = st.rs.border_admixture_score(
        sdata_new,
        min_transcripts=10**9,
        min_genes=10**9,
    )

    assert df["border_admixture_score"].isna().all()
    assert df["border_admixture_p_value"].isna().all()


def test_border_admixture_score_no_inplace(sdata_new):
    before_cols = set(sdata_new.tables["table"].obs.columns)

    df = st.rs.border_admixture_score(
        sdata_new,
        inplace=False,
    )

    after_cols = set(sdata_new.tables["table"].obs.columns)

    assert isinstance(df, pd.DataFrame)
    assert before_cols == after_cols


def test_border_admixture_score_invalid_n_permutations_raises(sdata_new):
    with pytest.raises(ValueError, match="n_permutations.*>= 100"):
        st.rs.border_admixture_score(
            sdata_new,
            n_permutations=99,
        )
