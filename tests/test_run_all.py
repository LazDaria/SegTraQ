import importlib.util
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geoseries_equal
from scipy import sparse
from spatialdata import SpatialData

import segtraq as st

st.settings.n_jobs = -1

# tolerances for comparing floating point outputs against the saved reference
RTOL = 1e-5
ATOL = 1e-8


def _load_generator():
    # tests are imported with `--import-mode=importlib`, so the script cannot be imported by name
    path = Path(__file__).parent / "generate_run_all_reference.py"
    spec = importlib.util.spec_from_file_location("generate_run_all_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_generator = _load_generator()
REFERENCE_PATH = _generator.REFERENCE_PATH
run_all_on_proseg = _generator.run_all_on_proseg


def _assert_array_equal(actual, expected, where):
    if sparse.issparse(actual) or sparse.issparse(expected):
        assert sparse.issparse(actual) and sparse.issparse(expected), f"{where}: sparse/dense mismatch"
        actual, expected = actual.toarray(), expected.toarray()
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, f"{where}: shape {actual.shape} != {expected.shape}"
    try:
        if np.issubdtype(actual.dtype, np.number) and np.issubdtype(expected.dtype, np.number):
            np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True)
        else:
            np.testing.assert_array_equal(actual, expected)
    except AssertionError as e:
        raise AssertionError(f"{where}: {e}") from e


def _assert_frame_equal(actual, expected, where):
    assert list(actual.columns) == list(expected.columns), (
        f"{where}: columns differ.\n"
        f"Only in result: {sorted(set(actual.columns) - set(expected.columns))}\n"
        f"Only in reference: {sorted(set(expected.columns) - set(actual.columns))}"
    )
    try:
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=RTOL, atol=ATOL)
    except AssertionError as e:
        raise AssertionError(f"{where}: {e}") from e


def _assert_equal(actual, expected, where):
    """Recursively compare (nested) containers, as stored in e.g. `.uns`."""
    if isinstance(expected, pd.DataFrame):
        assert isinstance(actual, pd.DataFrame), f"{where}: expected a DataFrame, got {type(actual)}"
        _assert_frame_equal(actual, expected, where)
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), f"{where}: expected a mapping, got {type(actual)}"
        assert set(actual.keys()) == set(expected.keys()), (
            f"{where}: keys differ.\n"
            f"Only in result: {sorted(set(actual.keys()) - set(expected.keys()))}\n"
            f"Only in reference: {sorted(set(expected.keys()) - set(actual.keys()))}"
        )
        for key in expected:
            _assert_equal(actual[key], expected[key], f"{where}[{key!r}]")
    elif isinstance(expected, str | bytes):
        assert actual == expected, f"{where}: {actual!r} != {expected!r}"
    else:
        _assert_array_equal(actual, expected, where)


def _assert_table_equal(actual, expected, where):
    assert actual.shape == expected.shape, f"{where}: shape {actual.shape} != {expected.shape}"
    _assert_frame_equal(actual.obs, expected.obs, f"{where}.obs")
    _assert_frame_equal(actual.var, expected.var, f"{where}.var")
    _assert_array_equal(actual.X, expected.X, f"{where}.X")
    for attr in ("layers", "obsm", "varm", "obsp", "varp", "uns"):
        _assert_equal(dict(getattr(actual, attr)), dict(getattr(expected, attr)), f"{where}.{attr}")


def assert_sdata_equal(actual: SpatialData, expected: SpatialData):
    """Assert that the tables, shapes and points of two SpatialData objects match."""
    for element_type in ("images", "labels", "points", "shapes", "tables"):
        assert set(getattr(actual, element_type).keys()) == set(getattr(expected, element_type).keys()), (
            f"{element_type} elements differ: "
            f"{sorted(getattr(actual, element_type).keys())} != {sorted(getattr(expected, element_type).keys())}"
        )

    for key, table in expected.tables.items():
        _assert_table_equal(actual.tables[key], table, f"tables[{key!r}]")

    for key, shapes in expected.shapes.items():
        where = f"shapes[{key!r}]"
        actual_shapes = actual.shapes[key]
        geometry_col = shapes.geometry.name
        _assert_frame_equal(
            pd.DataFrame(actual_shapes.drop(columns=geometry_col)),
            pd.DataFrame(shapes.drop(columns=geometry_col)),
            where,
        )
        try:
            assert_geoseries_equal(actual_shapes.geometry, shapes.geometry, check_less_precise=True)
        except AssertionError as e:
            raise AssertionError(f"{where}.geometry: {e}") from e

    for key, points in expected.points.items():
        _assert_frame_equal(actual.points[key].compute(), points.compute(), f"points[{key!r}]")


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


# runs every module (including the volume metrics) on the 3D ProSeg dataset and compares ALL outputs
# to a previously saved result, generated with `tests/generate_run_all_reference.py`.
# If a metric was changed on purpose, re-generate the reference with that script and upload it.
def test_run_all_matches_reference(tmp_path):
    assert REFERENCE_PATH.exists(), (
        f"Reference output {REFERENCE_PATH} not found. Download the latest test data "
        "or generate it with `python tests/generate_run_all_reference.py`."
    )

    result = run_all_on_proseg()

    # write and re-read the result, so that both objects went through the same zarr (de)serialization
    result.write(tmp_path / "run_all.zarr")
    actual = SpatialData.read(tmp_path / "run_all.zarr")
    expected = SpatialData.read(REFERENCE_PATH)

    assert_sdata_equal(actual, expected)
