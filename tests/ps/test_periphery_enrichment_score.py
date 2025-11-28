import pandas as pd

import segtraq as st


def test_periphery_enrichment_score(sdata_new):
    df = st.ps.periphery_enrichment_score(sdata_new, inplace=False)

    assert isinstance(df, pd.DataFrame), f"periphery_enrichment_score should return a DataFrame, got {type(df)}"
    expected_cols = {
        "center_expr",
        "border_expr",
        "cell_id",
        "center_area",
        "border_area",
        "border_density",
        "center_density",
        "density_ratio",
    }
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"
