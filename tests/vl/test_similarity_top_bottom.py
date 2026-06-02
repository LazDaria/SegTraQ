import pandas as pd
import pytest

import segtraq as st


def test_similarity_top_bottom_type(sdata_new):
    df = st.vl.similarity_top_bottom(sdata_new)

    assert isinstance(df, pd.DataFrame), f"compute_sim_top_bottom_z should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "similarity_top_bottom"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {df.columns}"


def test_similarity_top_bottom_values(sdata_new):
    df = st.vl.similarity_top_bottom(sdata_new)

    assert df["similarity_top_bottom"].dropna().between(-1, 1).all()


@pytest.mark.parametrize("q", [0.0, 0.5, -0.1, 0.9])
def test_similarity_top_bottom_invalid_q_raises(sdata_new, q):
    with pytest.raises(ValueError, match="q.*in \\(0, 0.5\\)"):
        st.vl.similarity_top_bottom(sdata_new, q=q)


def test_similarity_top_bottom_correct_z_drift_toggle_runs(sdata_new):
    df1 = st.vl.similarity_top_bottom(sdata_new, correct_z_drift=True)
    df2 = st.vl.similarity_top_bottom(sdata_new, correct_z_drift=False)

    assert isinstance(df1, pd.DataFrame)
    assert isinstance(df2, pd.DataFrame)
    assert set(df1.columns) == {"cell_id", "similarity_top_bottom"}
    assert set(df2.columns) == {"cell_id", "similarity_top_bottom"}


def test_similarity_top_bottom_inplace_writes_to_obs(sdata_new):
    key = "similarity_top_bottom"
    obs = sdata_new.tables["table"].obs
    if key in obs.columns:
        obs = obs.drop(columns=[key])
        sdata_new.tables["table"].obs = obs

    st.vl.similarity_top_bottom(sdata_new, inplace=True)
    assert key in sdata_new.tables["table"].obs.columns


def test_similarity_top_bottom_threshold_forces_nan(sdata_new):
    df = st.vl.similarity_top_bottom(sdata_new, min_transcripts=10**9)
    assert df["similarity_top_bottom"].isna().all()
