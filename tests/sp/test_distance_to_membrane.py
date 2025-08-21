import pandas as pd

import segtraq as st


def test_distance_to_membrane_structure(sdata_new):
    df = st.sp.distance_to_membrane(sdata_new, feature="LUM")

    assert isinstance(df, pd.DataFrame), f"centroid_mean_coord_diff should return a DataFrame, got {type(df)}"
    expected_cols = {
        "distance_to_outline_inverse",
        "distance_to_outline",
        "cell_area",
        "cell_id",
    }
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"


def test_distance_to_membrane_structure_values(sdata_new):
    df = st.sp.distance_to_membrane(sdata_new, feature="LUM")

    assert sum(df["distance_to_outline"] < df["cell_area"]) == df.shape[0], (
        "The distance metric is not always smaller than the area"
    )
