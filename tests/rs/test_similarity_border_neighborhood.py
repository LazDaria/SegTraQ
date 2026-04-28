import pandas as pd
import pytest

import segtraq as st

def test_similarity_border_neighborhood(sdata_new):
    df = st.rs.similarity_border_neighborhood(sdata_new)

    assert isinstance(df, pd.DataFrame), (
        "similarity_border_neighborhood should return a DataFrame, "
        f"got {type(df)}"
    )

    expected_cols = {
        "cell_id",
        "similarity_border_neighborhood",
    }

    assert set(df.columns) == expected_cols, (
        f"Expected columns {expected_cols}, but got {set(df.columns)}"
    )


def test_similarity_border_neighborhood_zero_radius(sdata_new):
    df = st.rs.similarity_border_neighborhood(
        sdata_new,
        neighborhood_radius_factor=0,
    )

    assert isinstance(df, pd.DataFrame)


def test_similarity_border_neighborhood_negative_radius(sdata_new):
    with pytest.raises(ValueError, match="`radius_factor` must be >= 0."):
        st.rs.similarity_border_neighborhood(
            sdata_new,
            neighborhood_radius_factor=-1,
        )

def test_similarity_border_neighborhood_negative_radius(sdata_new):
    with pytest.raises(ValueError, match="`radius_factor` must be >= 0."):
        st.rs.similarity_border_neighborhood(
            sdata_new,
            neighborhood_radius_factor=-1,
        )

def test_similarity_border_neighborhood_high_thresholds(sdata_new):
    df = st.rs.similarity_border_neighborhood(
        sdata_new,
        min_transcripts=10_000,
        min_genes=10_000,
    )

    assert df["similarity_border_neighborhood"].isna().all(), (
        "All similarities should be NaN when thresholds are too strict"
    )