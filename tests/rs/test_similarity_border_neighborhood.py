import pandas as pd
import pytest

import segtraq as st


def test_similarity_border_neighborhood(sdata_new):
    df = st.rs.similarity_border_neighborhood(sdata_new)
    assert isinstance(df, pd.DataFrame), "similarity_border_neighborhood should return a DataFrame, "
    f"got {type(df)}"
    expected_cols = {
        "similarity_center_border",
        "similarity_border_neighborhood",
        "ratio_border_neighborhood_to_center",
        "cell_id",
    }
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"


def test_similarity_border_neighborhood_no_cells(sdata_new):
    # no cells in the neighborhood
    with pytest.raises(
        AssertionError,
        match="`neighborhood_radius_factor` must be larger than 1.0.",
    ):
        st.rs.similarity_border_neighborhood(
            sdata_new,
            neighborhood_radius_factor=0,
        )


def test_similarity_border_neighborhood_no_erosion(sdata_new):
    # no erosion
    with pytest.raises(
        AssertionError,
        match="`erosion_fraction_of_radius` must be between 0 and 1",
    ):
        st.rs.similarity_border_neighborhood(
            sdata_new,
            erosion_fraction_of_radius=0,
        )


def test_similarity_border_neighborhood_negative_erosion(sdata_new):
    # negative erosion
    with pytest.raises(AssertionError, match="`erosion_fraction_of_radius` must be between 0 and 1"):
        st.rs.similarity_border_neighborhood(
            sdata_new,
            erosion_fraction_of_radius=-1,
        )


def test_similarity_border_neighborhood_negative_radius(sdata_new):
    # negative neighborhood radius
    with pytest.raises(AssertionError, match="`neighborhood_radius_factor` must be larger than 1"):
        st.rs.similarity_border_neighborhood(
            sdata_new,
            neighborhood_radius_factor=-1,
        )
