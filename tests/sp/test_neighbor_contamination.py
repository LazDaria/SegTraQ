import pandas as pd

import segtraq as st


def test_neighbor_contamination_realdata_inplace_outputs(sdata_labeled, adata_ref):
    markers = st.markers_from_reference(adata_ref.copy(), cell_type_key="celltype_major")

    per_cell_df, mat_df, bin_df = st.sp.neighbor_contamination(
        sdata=sdata_labeled,
        cell_type_key="transferred_cell_type",
        markers=markers,
        tables_key="table",
        tables_cell_id_key="cell_id",
        neighbors_key="spatial_connectivities",
        inplace=True,
    )

    assert isinstance(per_cell_df, pd.DataFrame)
    assert "negative_marker_contamination_counts" in per_cell_df.columns
    assert "negative_marker_contamination_fraction" in per_cell_df.columns

    assert isinstance(mat_df, pd.DataFrame)
    assert isinstance(bin_df, pd.DataFrame)

    # Inplace write checks
    tbl = sdata_labeled.tables["table"]
    assert "negative_marker_contamination" in tbl.uns
    assert "negative_marker_contamination_binary" in tbl.uns
    assert "negative_marker_contamination_counts" in tbl.obs.columns
    assert "negative_marker_contamination_fraction" in tbl.obs.columns
