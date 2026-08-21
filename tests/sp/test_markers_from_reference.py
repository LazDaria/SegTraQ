import pandas as pd

import segtraq as st

st.settings.n_jobs = -1


def test_markers_from_reference_real_adata_structure_and_overlap(adata_ref, markers):
    assert "celltype" in adata_ref.obs.columns
    n_types = adata_ref.obs["celltype"].nunique()
    assert n_types >= 2, "Need >= 2 types for differential markers"

    # Basic structure
    assert isinstance(markers, dict)
    assert set(markers.keys()) == set(pd.Categorical(adata_ref.obs["celltype"]).categories)
    for _ct, d in markers.items():
        assert set(d.keys()) == {"positive", "negative"}
        assert isinstance(d["positive"], list)
        assert isinstance(d["negative"], list)
        assert all(isinstance(g, str) for g in d["positive"])
        assert all(isinstance(g, str) for g in d["negative"])

    pos_all = [g for genes in (markers[ct]["positive"] for ct in markers) for g in genes]
    pos_counts = pd.Series(pos_all, dtype="object").value_counts()
    if len(pos_counts) > 0:
        assert (pos_counts <= (0.25 * n_types)).all(), (
            "Positive overlap filter failed: some genes appear in too many types"
        )
    neg_all = [g for genes in (markers[ct]["negative"] for ct in markers) for g in genes]
    neg_counts = pd.Series(neg_all, dtype="object").value_counts()
    if len(neg_counts) > 0:
        assert (neg_counts < n_types).all(), (
            "Negative overlap filter failed: a gene appears in all types' negative lists"
        )


def test_markers_from_reference_auc(adata_ref):
    n_types = adata_ref.obs["celltype"].nunique()
    markers_auc = st.markers_from_reference(
        adata_ref.copy(), ref_cell_type="celltype", t_pos=0.5, ref_raw_counts_layer="raw", mode="auc"
    )

    # Basic structure
    assert isinstance(markers_auc, dict)
    assert set(markers_auc.keys()) == set(pd.Categorical(adata_ref.obs["celltype"]).categories)
    for _ct, d in markers_auc.items():
        assert set(d.keys()) == {"positive", "negative"}
        assert isinstance(d["positive"], list)
        assert isinstance(d["negative"], list)
        assert all(isinstance(g, str) for g in d["positive"])
        assert all(isinstance(g, str) for g in d["negative"])

    pos_all = [g for genes in (markers_auc[ct]["positive"] for ct in markers_auc) for g in genes]
    pos_counts = pd.Series(pos_all, dtype="object").value_counts()
    if len(pos_counts) > 0:
        assert (pos_counts <= (0.25 * n_types)).all(), (
            "Positive overlap filter failed: some genes appear in too many types"
        )
    neg_all = [g for genes in (markers_auc[ct]["negative"] for ct in markers_auc) for g in genes]
    neg_counts = pd.Series(neg_all, dtype="object").value_counts()
    if len(neg_counts) > 0:
        assert (neg_counts < n_types).all(), (
            "Negative overlap filter failed: a gene appears in all types' negative lists"
        )


def test_overlap_filter_effect_without_internals(adata_ref):
    markers_loose = st.markers_from_reference(
        adata_ref.copy(), ref_cell_type="celltype", t_pos=0.5, ref_raw_counts_layer="raw"
    )
    markers_strict = st.markers_from_reference(
        adata_ref.copy(), ref_cell_type="celltype", t_pos=0.1, ref_raw_counts_layer="raw"
    )

    n_types = adata_ref.obs["celltype"].nunique()

    def count_across_types(marker_dict, key="positive"):
        all_genes = [g for ct in marker_dict for g in marker_dict[ct][key]]
        return pd.Series(all_genes, dtype="object").value_counts()

    pos_counts_loose = count_across_types(markers_loose, key="positive")
    pos_counts_strict = count_across_types(markers_strict, key="positive")

    # 1) Contract checks: outputs must respect their own thresholds
    if not pos_counts_loose.empty:
        assert (pos_counts_loose < (0.5 * n_types)).all(), "Loose markers violate t_pos=0.5 overlap contract"
    if not pos_counts_strict.empty:
        assert (pos_counts_strict < (0.1 * n_types)).all(), "Strict markers violate t_pos=0.1 overlap contract"

    if not pos_counts_loose.empty and not pos_counts_strict.empty:
        assert pos_counts_strict.max() <= pos_counts_loose.max()

    for ct in markers_loose:
        loose_set = set(markers_loose[ct]["positive"])
        strict_set = set(markers_strict[ct]["positive"])
        assert strict_set.issubset(loose_set), f"{ct}: strict not subset of loose"
