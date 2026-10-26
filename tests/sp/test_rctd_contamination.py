import pandas as pd

import segtraq


def test_rctd_contamination_realdata_runs_and_stores_in_obs(
    sdata_labeled,
    adata_ref,
):
    df = segtraq.sp.rctd_contamination(
        sdata=sdata_labeled,
        adata_ref=adata_ref,
        ref_cell_type="celltype",
        ref_raw_counts_layer="raw",
        tables_key="table",
        inplace=True,
    )

    assert isinstance(df, pd.DataFrame)
    assert {"cell_id", "rctd_contamination"}.issubset(df.columns)

    adata = sdata_labeled.tables["table"]
    assert len(df) == adata.n_obs
    assert "rctd_contamination" in adata.obs

    # Valid scores must be between 0 and 1.
    scores = df["rctd_contamination"].dropna()
    assert len(scores) > 0
    assert scores.between(0, 1).all()

    # Check that results are stored in-place.
    pd.testing.assert_series_equal(
        adata.obs.set_index("cell_id")["rctd_contamination"].sort_index(),
        df.set_index("cell_id")["rctd_contamination"].sort_index(),
        check_names=False,
    )


def test_rctd_contamination_realdata_no_inplace(
    sdata_labeled,
    adata_ref,
):
    adata = sdata_labeled.tables["table"]
    obs_before = adata.obs.copy(deep=True)

    df = segtraq.sp.rctd_contamination(
        sdata=sdata_labeled,
        adata_ref=adata_ref,
        ref_cell_type="celltype",
        ref_raw_counts_layer="raw",
        tables_key="table",
        inplace=False,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == adata.n_obs
    assert df["cell_id"].tolist() == adata.obs["cell_id"].tolist()
    assert df["rctd_contamination"].notna().any()

    # Neither add columns nor overwrite existing values.
    pd.testing.assert_frame_equal(adata.obs, obs_before)
