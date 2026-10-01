import pytest

import segtraq as st

st.settings.n_jobs = -1


# this only tests that run_all works without errors and that it correctly determines
# which modules can and cannot be run, given the arguments it was passed.
def test_run_all_skips_modules_missing_prerequisites(segtraq_obj):
    with pytest.warns(UserWarning, match="run_supervised"):
        result = segtraq_obj.run_all(inplace=False)

    # supervised metrics require a reference dataset;
    # none is provided here, so this module should be skipped automatically
    assert "supervised" in result["skipped"]
    assert result["supervised"] is None

    # all other modules do not strictly require a reference and should run successfully
    assert result["baseline"] is not None
    assert "num_cells" in result["baseline"]

    assert result["region_similarity"] is not None
    assert "ious" in result["region_similarity"]

    assert result["volume"] is not None
    assert "similarity_top_bottom" in result["volume"]

    assert result["clustering_stability"] is not None
    assert "cluster_connectedness" in result["clustering_stability"]

    assert result["point_statistics"] is not None
    assert "distance_to_centroid" in result["point_statistics"]

    for name in ("baseline", "region_similarity", "volume", "clustering_stability", "point_statistics"):
        assert name not in result["skipped"]


# TODO: this should be replaced with a test that compares ALL outputs to a previously saved result
# given a reference, cell type key, and markers, every module should run
def test_run_all_runs_every_module_when_prerequisites_are_met(
    segtraq_obj,
    markers,
    adata_ref,
):
    result = segtraq_obj.run_all(
        adata_ref=adata_ref,
        ref_cell_type="celltype",
        ref_raw_counts_layer="raw",
        cell_type_key="transferred_cell_type",
        markers=markers,
        inplace=False,
    )

    assert result["skipped"] == {}

    for name in (
        "baseline",
        "region_similarity",
        "volume",
        "clustering_stability",
        "supervised",
        "point_statistics",
    ):
        assert result[name] is not None

    assert "marker_balanced_accuracy" in result["supervised"]["marker_purity"].columns
