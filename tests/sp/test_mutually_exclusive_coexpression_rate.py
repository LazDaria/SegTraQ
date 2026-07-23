import pandas as pd

import segtraq


def test_mecr_realdata_runs_and_stores_in_uns(sdata_3D_labeled, adata_ref):
    # subsetting to commong genes
    adata = sdata_3D_labeled.tables["table"]
    common_genes = adata.var_names[adata.var_names.isin(adata_ref.var_names)]
    adata_ref = adata_ref[:, common_genes].copy()

    # markers from your existing reference fixture
    markers = segtraq.markers_from_reference(
        adata_ref.copy(),
        ref_cell_type="celltype",
        pval_adj_thresh=1.0,  # default: 0.05
        logfc_pos_thresh=0.0,  # default: 1.0
        vote_fraction_pos=0.0,  # default: 0.5
        min_pos_frac=0.1,  # default: 0.1
        max_neg_frac=0.05,  # default: 0.05
        t_pos=1.0,  # default: 0.25 (this was the culprit, setting this to 0 has the opposite effect of what I wanted)
        t_neg=1.0,  # default: 1.0
        min_cells_per_celltype=1,  # default: 10
        ref_raw_counts_layer="raw",  # default: None
    )

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
