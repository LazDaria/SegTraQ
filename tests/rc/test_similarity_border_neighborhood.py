import pandas as pd

import segtraq as st


def test_similarity_border_neighborhood(sdata_new):
    df = st.rc.similarity_border_neighborhood(sdata_new)
    assert isinstance(df, pd.DataFrame), "similarity_border_neighborhood should return a DataFrame, "
    f"got {type(df)}"
    expected_cols = {
        "similarity_center_border",
        "similarity_border_neighborhood",
        "ratio_border_neighborhood_to_center",
        "cell_id",
    }
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"
