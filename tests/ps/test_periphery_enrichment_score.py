import pandas as pd

import segtraq as st


def test_periphery_enrichment_score(sdata_new):
    df = st.ps.periphery_enrichment_score(sdata_new, inplace=False)

    assert isinstance(df, pd.DataFrame), f"periphery_enrichment_score should return a DataFrame, got {type(df)}"
    expected_cols = {
        "center_expr_all_genes",
        "border_expr_all_genes",
        "cell_id",
        "center_area_all_genes",
        "border_area_all_genes",
        "border_density_all_genes",
        "center_density_all_genes",
        "density_ratio_all_genes",
    }
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {set(df.columns)}"
