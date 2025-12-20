import pandas as pd

import segtraq as st


def test_centroid_mean_coord_diff_structure(sdata_new):
    df = st.ps.centroid_mean_coord_diff(sdata_new, genes="LUM", inplace=False)

    assert isinstance(df, pd.DataFrame), f"centroid_mean_coord_diff should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "x_cell", "x", "y_cell", "y", "distance", "cell_area", "distance_LUM"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {df.columns}"
