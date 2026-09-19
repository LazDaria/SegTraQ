import pytest

import segtraq as st

st.settings.n_jobs = -1


# this only tests that run_all works without errors and that it correctly determines
# which modules can and cannot be run, given the arguments it was passed.
def test_run_all_skips_modules_missing_prerequisites(segtraq_obj):
    with pytest.warns(UserWarning, match="run_supervised"):
        result = segtraq_obj.run_all(inplace=False)

    # supervised metrics require either `cell_type_key`+`markers` or a reference dataset;
    # none of these are provided here, so this module should be skipped automatically
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


# given a cell type key and markers, every module (including supervised) should run
def test_run_all_runs_every_module_when_prerequisites_are_met(segtraq_obj, markers):
    result = segtraq_obj.run_all(
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


# inplace=True should merge results into the underlying sdata object and return None
def test_run_all_inplace_writes_into_sdata(segtraq_obj, markers):
    result = segtraq_obj.run_all(
        cell_type_key="transferred_cell_type",
        markers=markers,
        inplace=True,
    )

    assert result is None

    obs = segtraq_obj.sdata.tables["table"].obs
    uns = segtraq_obj.sdata.tables["table"].uns

    assert "num_cells" in uns
    assert "iou" in obs.columns
    assert "similarity_top_bottom" in obs.columns
