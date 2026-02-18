import pandas as pd
import pytest

import segtraq as st


def test_marker_purity_weight_bounds_raises(sdata_labeled, adata_ref):
    markers = st.markers_from_reference(adata_ref.copy(), cell_type_key="celltype_major")

    with pytest.raises(ValueError, match="weight_cont must be between 0 and 1"):
        st.sp.marker_purity(
            sdata=sdata_labeled,
            cell_type_key="transferred_cell_type",
            markers=markers,
            weight_cont=1.1,
            neighbors_key="spatial_connectivities",
            inplace=False,
        )


def test_marker_purity_realdata_inplace_writes_obs(sdata_labeled, adata_ref):
    markers = st.markers_from_reference(adata_ref.copy(), cell_type_key="celltype_major")

    df = st.sp.marker_purity(
        sdata=sdata_labeled,
        cell_type_key="transferred_cell_type",
        markers=markers,
        neighbors_key="spatial_connectivities",
        inplace=True,
    )

    assert isinstance(df, pd.DataFrame)
    expected_cols = {
        "positive_precision",
        "positive_recall",
        "positive_F1",
        "negative_precision",
        "negative_recall",
        "negative_F1",
        "F1_purity",
    }
    assert expected_cols.issubset(df.columns)

    # Inplace columns present
    for c in expected_cols:
        assert c in sdata_labeled.tables["table"].obs.columns
