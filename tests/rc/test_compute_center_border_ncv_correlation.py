import pandas as pd

import segtraq as st


def test_compute_center_border_ncv_correlation(sdata_new):
    # import pdb; pdb.set_trace()
    df = st.rc.compute_center_border_ncv_correlation(sdata_new)  # TODO: SOMETHING IS WRONG HERE WITH THE IDS
    assert isinstance(df, pd.DataFrame), "compute_center_border_ncv_correlation should return a DataFrame, "
    f"got {type(df)}"
    expected_cols = {"corr_ncv_vs_center", "corr_border_ncv", "corr_center_border", "cell_id"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"
