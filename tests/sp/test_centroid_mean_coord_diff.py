import pandas as pd

import segtraq as st


def test_centroid_mean_coord_diff_structure(sdata_new):
    df = st.sp.centroid_mean_coord_diff(sdata_new, feature="LUM")

    assert isinstance(df, pd.DataFrame), f"centroid_mean_coord_diff should return a DataFrame, got {type(df)}"
    expected_cols = {
        "cell_id",
        "centroid_x",
        "centroid_y",
        "x",
        "y",
        "distance",
        "cell_area",
    }
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"


def test_centroid_mean_coord_diff_values(sdata_new):
    df = st.sp.centroid_mean_coord_diff(sdata_new, feature="LUM")

    assert sum(df["distance"] < df["cell_area"]) == df.shape[0], (
        "The distance metric is not always smaller than the area"
    )
