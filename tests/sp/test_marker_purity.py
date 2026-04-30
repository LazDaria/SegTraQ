import pandas as pd

import segtraq as st


def test_marker_purity_values_are_in_unit_interval(sdata_labeled, adata_ref):
    markers = st.markers_from_reference(adata_ref.copy(), cell_type_key="celltype_major")

    df = st.sp.marker_purity(
        sdata=sdata_labeled,
        cell_type_key="transferred_cell_type",
        markers=markers,
        neighbors_key="spatial_connectivities",
        inplace=False,
    )

    metric_cols = [
        "positive_marker_recall",
        "negative_marker_avoidance",
        "marker_balanced_accuracy",
    ]

    for col in metric_cols:
        vals = df[col].dropna()
        assert ((vals >= 0) & (vals <= 1)).all()


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
        "positive_marker_recall",
        "negative_marker_avoidance",
        "marker_balanced_accuracy",
        "n_evaluated_positive_markers",
        "n_evaluated_negative_markers",
    }

    assert expected_cols.issubset(df.columns)

    for c in expected_cols:
        assert c in sdata_labeled.tables["table"].obs.columns
