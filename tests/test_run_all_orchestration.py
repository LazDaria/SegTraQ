from unittest.mock import patch

import segtraq as st


def _bare_segtraq():
    """A SegTraQ instance without going through __init__ (no real SpatialData needed).

    `run_all` only calls other `SegTraQ` methods (which are patched below), so no
    attributes besides the mocked methods themselves are required for these tests.
    """
    return st.SegTraQ.__new__(st.SegTraQ)


def _patch_runners(**overrides):
    """Patch every module-runner used by `run_all` with a MagicMock, applying `overrides`."""
    targets = [
        "run_baseline",
        "run_region_similarity",
        "run_volume",
        "run_clustering_stability",
        "run_supervised",
        "run_point_statistics",
        "run_label_transfer",
    ]
    patchers = {name: patch.object(st.SegTraQ, name) for name in targets}
    mocks = {name: patcher.start() for name, patcher in patchers.items()}
    for name, side_effect in overrides.items():
        mocks[name].side_effect = side_effect
    return mocks, patchers


def _stop_all(patchers):
    for patcher in patchers.values():
        patcher.stop()


def test_run_all_calls_every_module_runner():
    mocks, patchers = _patch_runners()
    try:
        segtraq = _bare_segtraq()
        segtraq.run_all(inplace=True)
    finally:
        _stop_all(patchers)

    for name in (
        "run_baseline",
        "run_region_similarity",
        "run_volume",
        "run_clustering_stability",
        "run_supervised",
        "run_point_statistics",
    ):
        mocks[name].assert_called_once()

    # no reference dataset was provided, so label transfer should never be attempted
    mocks["run_label_transfer"].assert_not_called()


def test_run_all_forwards_inplace_and_module_kwargs():
    mocks, patchers = _patch_runners()
    try:
        segtraq = _bare_segtraq()
        segtraq.run_all(
            inplace=False,
            baseline_kwargs={"morphological_kwargs": {"n_jobs": 2}},
            region_similarity_kwargs={"n_jobs": 4},
            clustering_stability_kwargs={"key_prefix": "custom_prefix"},
        )
    finally:
        _stop_all(patchers)

    # `_leave=False` is added by `run_all` itself so nested progress bars are cleared on completion
    mocks["run_baseline"].assert_called_once_with(inplace=False, _leave=False, morphological_kwargs={"n_jobs": 2})
    mocks["run_region_similarity"].assert_called_once_with(inplace=False, _leave=False, n_jobs=4)
    mocks["run_clustering_stability"].assert_called_once_with(inplace=False, _leave=False, key_prefix="custom_prefix")


def test_run_all_continues_and_records_skips_when_a_module_raises():
    def raise_for_volume(*args, **kwargs):
        raise AssertionError("Cannot run volume metrics for 2D data")

    mocks, patchers = _patch_runners(run_volume=raise_for_volume)
    try:
        segtraq = _bare_segtraq()
        result = segtraq.run_all(inplace=False)
    finally:
        _stop_all(patchers)

    # the failing module is recorded as skipped, with the reason it failed
    assert "volume" in result["skipped"]
    assert "2D data" in result["skipped"]["volume"]
    assert result["volume"] is None

    # every other module still ran despite `run_volume` raising
    for name in ("run_baseline", "run_region_similarity", "run_clustering_stability", "run_supervised"):
        mocks[name].assert_called_once()
    assert result["skipped"].keys() == {"volume"}


def test_run_all_returns_none_when_inplace():
    mocks, patchers = _patch_runners()
    try:
        segtraq = _bare_segtraq()
        result = segtraq.run_all(inplace=True)
    finally:
        _stop_all(patchers)

    assert result is None


def test_run_all_runs_label_transfer_once_and_reuses_it():
    adata_ref = object()

    mocks, patchers = _patch_runners()
    try:
        segtraq = _bare_segtraq()
        segtraq.run_all(
            adata_ref=adata_ref,
            ref_cell_type="celltype",
            inplace=True,
        )
    finally:
        _stop_all(patchers)

    # label transfer is run exactly once upfront, not once per downstream module
    mocks["run_label_transfer"].assert_called_once_with(
        adata_ref=adata_ref,
        ref_cell_type="celltype",
        ref_gene_key=None,
        query_gene_key=None,
        ref_raw_counts_layer=None,
        cell_type_key="transferred_cell_type",
        inplace=True,
    )

    # its result ("transferred_cell_type") is then forwarded to every module that accepts
    # a `cell_type_key`, instead of letting each of them recompute label transfer on its own
    for name in ("run_volume", "run_supervised"):
        _, kwargs = mocks[name].call_args
        assert kwargs["cell_type_key"] == "transferred_cell_type"
        assert kwargs["adata_ref"] is adata_ref

    _, ps_kwargs = mocks["run_point_statistics"].call_args
    assert ps_kwargs["cell_type_key"] == "transferred_cell_type"


def test_run_all_skips_label_transfer_when_cell_type_key_given():
    mocks, patchers = _patch_runners()
    try:
        segtraq = _bare_segtraq()
        segtraq.run_all(
            adata_ref=object(),
            ref_cell_type="celltype",
            cell_type_key="already_known_cell_type",
            inplace=True,
        )
    finally:
        _stop_all(patchers)

    mocks["run_label_transfer"].assert_not_called()

    _, kwargs = mocks["run_volume"].call_args
    assert kwargs["cell_type_key"] == "already_known_cell_type"


def test_run_all_falls_back_gracefully_when_label_transfer_fails():
    mocks, patchers = _patch_runners(run_label_transfer=ValueError("no matching genes"))
    try:
        segtraq = _bare_segtraq()
        result = segtraq.run_all(
            adata_ref=object(),
            ref_cell_type="celltype",
            inplace=False,
        )
    finally:
        _stop_all(patchers)

    # label transfer itself is not one of the six metric modules, so it is not reported
    # under `skipped`, but downstream modules should fall back to no `cell_type_key`
    assert "label_transfer" not in result["skipped"]
    _, kwargs = mocks["run_volume"].call_args
    assert kwargs["cell_type_key"] is None


def test_run_all_result_keys():
    mocks, patchers = _patch_runners()
    try:
        segtraq = _bare_segtraq()
        result = segtraq.run_all(inplace=False)
    finally:
        _stop_all(patchers)

    assert set(result.keys()) == {
        "baseline",
        "region_similarity",
        "volume",
        "clustering_stability",
        "supervised",
        "point_statistics",
        "skipped",
    }
