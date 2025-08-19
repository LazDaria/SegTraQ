import pandas as pd

import segtraq as st


def test_shape_metrics_structure(sdata_new):
    df = st.tm.shape_metrics(sdata_new)

    assert isinstance(df, pd.DataFrame), f"shape_metrics should return a DataFrame, got {type(df)}"
    expected_cols = {"mean", "variance", "skew", "kurtosis"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"
