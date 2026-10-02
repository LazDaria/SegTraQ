import pandas as pd

import segtraq


def test_mecr_realdata_runs_and_stores_in_uns(
    sdata_3D_labeled,
    adata_ref,
    markers,
):
    df = segtraq.sp.mutually_exclusive_coexpression_rate(
        sdata=sdata_3D_labeled,
        adata_ref=adata_ref,
        ref_cell_type="celltype",
        ref_raw_counts_layer="raw",
        markers=markers,
        tables_key="table",
        inplace=True,
    )

    assert isinstance(df, pd.DataFrame)

    expected_columns = {
        "gene1",
        "gene2",
        "odds_ratio",
        "pvalue",
        "pvalue_adj",
        "a",
        "b",
        "c",
        "d",
    }
    assert expected_columns.issubset(df.columns), (
        f"Expected columns not found in the result DataFrame. Found columns: {df.columns}"
    )

    # Check that the results are stored in-place.
    adata = sdata_3D_labeled.tables["table"]
    assert "mutually_exclusive_coexpression_rate" in adata.uns
    assert adata.uns["mutually_exclusive_coexpression_rate"].equals(df)
