import pandas as pd

import segtraq as st


def test_distance_to_membrane_structure(sdata_new):
    df = st.ps.distance_to_membrane(sdata_new, genes="LUM", inplace=False)

    assert isinstance(df, pd.DataFrame), f"centroid_mean_coord_diff should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "distance_to_outline_aggregated_LUM", "cell_area", "distance_to_outline_norm_aggregated_LUM", "distance_to_outline_inverse_aggregated_LUM"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"
