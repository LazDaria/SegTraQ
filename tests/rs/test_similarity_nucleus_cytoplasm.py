import numpy as np
import pandas as pd

import segtraq as st


def test_similarity_nucleus_cytoplasm(sdata_new):
    df = st.rs.similarity_nucleus_cytoplasm(sdata_new)
    # identify cells with missing nucleus_id
    mask = df["nucleus_id"].isna()

    # test that there is nothing computed for a cell without a nucleus
    for cell_id in df.loc[mask, "cell_id"]:
        corr = df.loc[df["cell_id"] == cell_id, "similarity_nucleus_cytoplasm"].iloc[0]
        assert np.isnan(corr), f"Expected NaN for cell {cell_id} with missing nucleus, got {corr}"

    # test that there is a valid correlation for cells with nucleus
    assert isinstance(df, pd.DataFrame), f"similarity_nucleus_cytoplasm should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "nucleus_id", "iou", "similarity_nucleus_cytoplasm", "nucleus_fraction"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"
