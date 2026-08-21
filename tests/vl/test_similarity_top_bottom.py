import pandas as pd
import pytest

import segtraq as st

st.settings.n_jobs = -1


def test_similarity_top_bottom_type(sdata_new):
    df = st.vl.similarity_top_bottom(sdata_new)

    assert isinstance(df, pd.DataFrame), f"similarity_top_bottom should return a DataFrame, got {type(df)}"

    expected_cols = {
        "cell_id",
        "similarity_top_bottom",
        "similarity_top_bottom_p_value",
    }

    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"

    assert pd.api.types.is_float_dtype(df["similarity_top_bottom"])
    assert pd.api.types.is_float_dtype(df["similarity_top_bottom_p_value"])


def test_similarity_top_bottom_values(sdata_new):
    df = st.vl.similarity_top_bottom(sdata_new)

    # Residual = observed cosine similarity - mean null similarity
    vals = df["similarity_top_bottom"].dropna()
    assert vals.between(-2, 2).all()

    pvals = df["similarity_top_bottom_p_value"].dropna()
    assert pvals.between(0, 1).all()


@pytest.mark.parametrize("q", [0.0, 0.5, -0.1, 0.9])
def test_similarity_top_bottom_invalid_q_raises(sdata_new, q):
    with pytest.raises(ValueError, match=r"q.*in \(0, 0.5\)"):
        st.vl.similarity_top_bottom(sdata_new, q=q)


def test_similarity_top_bottom_correct_z_drift_toggle_runs(sdata_new):
    df1 = st.vl.similarity_top_bottom(
        sdata_new,
        correct_z_drift=True,
    )
    df2 = st.vl.similarity_top_bottom(
        sdata_new,
        correct_z_drift=False,
    )

    expected_cols = {
        "cell_id",
        "similarity_top_bottom",
        "similarity_top_bottom_p_value",
    }

    assert isinstance(df1, pd.DataFrame)
    assert isinstance(df2, pd.DataFrame)
    assert set(df1.columns) == expected_cols
    assert set(df2.columns) == expected_cols


def test_similarity_top_bottom_inplace_writes_to_obs(sdata_new):
    keys = [
        "similarity_top_bottom",
        "similarity_top_bottom_p_value",
    ]

    obs = sdata_new.tables["table"].obs

    existing = [key for key in keys if key in obs.columns]

    if existing:
        obs = obs.drop(columns=existing)
        sdata_new.tables["table"].obs = obs

    st.vl.similarity_top_bottom(
        sdata_new,
        inplace=True,
    )

    for key in keys:
        assert key in sdata_new.tables["table"].obs.columns


def test_similarity_top_bottom_threshold_forces_nan(sdata_new):
    df = st.vl.similarity_top_bottom(
        sdata_new,
        min_transcripts=10**9,
    )

    assert df["similarity_top_bottom"].isna().all()
    assert df["similarity_top_bottom_p_value"].isna().all()
