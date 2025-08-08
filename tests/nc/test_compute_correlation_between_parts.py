import numpy as np
import pandas as pd

import segtraq as st


def test_missing_nucleus(sdata_new):
    df = st.nc.compute_correlation_between_parts(sdata_new)
    # identify cells with missing best_nuc_id
    mask = df["best_nuc_id"].isna()
    for cell_id in df.loc[mask, "cell_id"]:
        corr = df.loc[df["cell_id"] == cell_id, "correlation_parts"].iloc[0]
        assert np.isnan(corr), f"Expected NaN for cell {cell_id} with missing nucleus, got {corr}"


def test_output_integrity(sdata_new):
    df = st.nc.compute_correlation_between_parts(sdata_new)
    assert isinstance(df, pd.DataFrame), f"compute_correlation_between_parts should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "best_nuc_id", "IoU", "correlation_parts"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"
