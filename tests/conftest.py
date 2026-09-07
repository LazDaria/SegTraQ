from pathlib import Path

import anndata as ad
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from spatialdata import SpatialData

import segtraq as st

st.settings.n_jobs = -1


@pytest.fixture(scope="session", name="sdata_new")
def test_sdata_new():
    """Load the SpatialData test sample once per test session."""

    test_data_path = Path(__file__).parent / "data" / "xenium.zarr"
    sdata_new = SpatialData.read(test_data_path)
    st.validate_spatialdata(sdata_new, images_key="image", tables_centroid_x_key=None, tables_centroid_y_key=None)

    # subsetting for faster tests
    bb_xmin = 800
    bb_ymin = 1150
    bb_w = 200
    bb_h = 300
    bb_xmax = bb_xmin + bb_w
    bb_ymax = bb_ymin + bb_h

    sdata_new = sdata_new.query.bounding_box(
        axes=["x", "y"],
        min_coordinate=[bb_xmin, bb_ymin],
        max_coordinate=[bb_xmax, bb_ymax],
        target_coordinate_system="global",
    )

    # to catch issues in any merging code, we have the index and a column contain the cell IDs
    sdata_new.tables["table"].obs.index = sdata_new.tables["table"].obs["cell_id"].values
    sdata_new.tables["table"].obs.index.name = "cell_id"

    # this is important, because the test object initially contains some duplicate nucleus_ids
    # by calling validate_spatialdata,
    # we ensure that these get resolved before continuing with the tests
    st.validate_spatialdata(sdata_new, images_key="image", tables_centroid_x_key=None, tables_centroid_y_key=None)

    return sdata_new


@pytest.fixture(scope="session", name="sdata_3D")
def test_sdata_3D():
    """Load the SpatialData test sample once per test session."""

    test_data_path = Path(__file__).parent / "data" / "proseg2.zarr"
    sdata_3D = SpatialData.read(test_data_path)
    st.SegTraQ(
        sdata_3D,
        points_cell_id_key="assignment",
        points_background_id=None,
        points_gene_key="gene",
        tables_area_key="volume",
        tables_cell_id_key="cell",
        shapes_cell_id_key="cell",
        tables_centroid_x_key="centroid_x",
        tables_centroid_y_key="centroid_y",
        filter_kwargs={"inplace": False},
    )

    return sdata_3D


@pytest.fixture(scope="session", name="adata_ref")
def test_adata_ref():
    """Load the AnnData reference sample once per test session."""

    test_data_path = Path(__file__).parent / "data" / "scRNAseq_ref_subset.h5ad"
    adata_ref = ad.read_h5ad(test_data_path)
    return adata_ref


@pytest.fixture(scope="session", name="sdata_labeled")
def test_sdata_labeled(sdata_new, adata_ref):
    # run label transfer once; modifies sdata_new in place
    st.run_label_transfer(
        sdata=sdata_new,
        adata_ref=adata_ref,
        ref_cell_type="celltype",
        ref_raw_counts_layer="raw",
        inplace=True,
    )
    return sdata_new


@pytest.fixture(scope="session", name="sdata_3D_labeled")
def test_sdata_3D_labeled(sdata_3D, adata_ref):
    # run label transfer once; modifies sdata_new in place
    st.run_label_transfer(
        sdata=sdata_3D,
        adata_ref=adata_ref,
        ref_cell_type="celltype",
        tables_cell_id_key="cell",
        points_key="transcripts",
        points_cell_id_key="assignment",
        points_gene_key="gene",
        ref_raw_counts_layer="raw",
        inplace=True,
    )
    return sdata_3D


@pytest.fixture(scope="session", name="segtraq_obj")
def test_segtraq_obj(sdata_labeled):
    """Load the SpatialData test sample once per test session."""
    sdata = sd.deepcopy(sdata_labeled)  # to avoid modifying the original sdata_labeled in place
    # to make this more difficult, we rename the cell column in the shapes
    # this should flag issues from mismatching IDs between the tables and shapes
    sdata.shapes["cell_boundaries"].index.name = "cell_id_1"
    # we also rename the cell_id column in the tables
    # in reality, sdata objects should rarely be this inconsistent
    # but this allows us to test that the segtraq object can still be created as long as the correct keys are provided
    sdata.tables["table"].obs = sdata.tables["table"].obs.rename(columns={"cell_id": "cell_id_2"})
    # to catch issues in any merging code, we also make the index of the obs contain the cell IDs
    sdata.tables["table"].obs.index = sdata.tables["table"].obs["cell_id_2"].values
    sdata.tables["table"].obs.index.name = "cell_id_2"
    sdata.tables["table"].uns["spatialdata_attrs"]["instance_key"] = "cell_id_2"
    # creating a segtraq object
    return st.SegTraQ(
        sdata,
        tables_centroid_x_key="x_centroid",
        tables_centroid_y_key="y_centroid",
        images_key="image",
        shapes_cell_id_key="cell_id_1",
        tables_cell_id_key="cell_id_2",
        filter_kwargs={"inplace": False},
    )


@pytest.fixture(scope="session", name="markers")
def test_markers(adata_ref):
    return st.markers_from_reference(
        adata_ref.copy(),
        ref_cell_type="celltype",
        ref_raw_counts_layer="raw",
    )


def _normalize_snapshot_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.floating, float)):
        return round(float(value), 8)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return str(value)


def _df_snapshot(df: pd.DataFrame):
    records = []
    for record in df.to_dict(orient="records"):
        records.append({str(k): _normalize_snapshot_value(v) for k, v in record.items()})
    records.sort(key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")))
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "columns": sorted([str(c) for c in df.columns]),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def _scalar_snapshot(value):
    return _normalize_snapshot_value(value)


def _markers_snapshot(markers):
    normalized = {}
    for cell_type, marker_dict in markers.items():
        normalized[str(cell_type)] = {
            "positive": sorted([str(g) for g in marker_dict["positive"]]),
            "negative": sorted([str(g) for g in marker_dict["negative"]]),
        }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return {
        "cell_types": sorted(list(normalized.keys())),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def _axes_snapshot(axes):
    flat_axes = []
    for item in axes:
        if isinstance(item, list):
            flat_axes.extend(item)
        else:
            flat_axes.append(item)

    snapshots = []
    for ax in flat_axes:
        lines = []
        for line in ax.lines:
            x = [_normalize_snapshot_value(v) for v in line.get_xdata()]
            y = [_normalize_snapshot_value(v) for v in line.get_ydata()]
            lines.append({"x": x, "y": y})

        snapshots.append(
            {
                "title": ax.get_title(),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "xlim": [_normalize_snapshot_value(v) for v in ax.get_xlim()],
                "ylim": [_normalize_snapshot_value(v) for v in ax.get_ylim()],
                "line_count": len(ax.lines),
                "sha256": hashlib.sha256(json.dumps(lines, sort_keys=True).encode()).hexdigest(),
            }
        )

    snapshots.sort(key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")))
    return snapshots


@pytest.fixture(scope="session", name="snapshot_helpers")
def test_snapshot_helpers():
    return {
        "scalar": _scalar_snapshot,
        "df": _df_snapshot,
        "markers": _markers_snapshot,
        "axes": _axes_snapshot,
    }
