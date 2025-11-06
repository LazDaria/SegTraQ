import pandas as pd

import segtraq as st


def test_distance_to_membrane_structure(sdata_new):
    df = st.ps.distance_to_membrane(sdata_new, feature="LUM", inplace=False)

    assert isinstance(df, pd.DataFrame), f"centroid_mean_coord_diff should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_area", "distance_to_outline_inverse_LUM", "cell_id", "distance_to_outline_LUM"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"
