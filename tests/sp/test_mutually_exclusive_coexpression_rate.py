import pandas as pd

import segtraq


def test_mecr_realdata_runs_and_stores_in_uns(sdata_3D_labeled, markers):
    df = segtraq.sp.mutually_exclusive_coexpression_rate(
        sdata=sdata_3D_labeled,
        markers=markers,
        tables_key="table",
        inplace=True,
    )

    assert isinstance(df, pd.DataFrame)
    assert set(["gene1", "gene2", "odds_ratio", "pvalue", "a", "b", "c", "d"]).issubset(df.columns), (
        f"Expected columns not found in the result DataFrame. Found columns: {df.columns}"
    )

    # check that the results are stored in-place
    adata = sdata_3D_labeled.tables["table"]
    assert "mutually_exclusive_coexpression_rate" in adata.uns
    assert adata.uns["mutually_exclusive_coexpression_rate"].equals(df)
