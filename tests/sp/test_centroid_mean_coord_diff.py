import pandas as pd

import segtraq as st


def test_centroid_mean_coord_diff_structure(sdata_new):
    df = st.ps.centroid_mean_coord_diff(sdata_new, genes="LUM", inplace=False)

    assert isinstance(df, pd.DataFrame), f"centroid_mean_coord_diff should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "centroid_y", "distance_LUM", "y", "cell_area", "centroid_x", "x", "distance"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"
