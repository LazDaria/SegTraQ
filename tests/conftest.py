from pathlib import Path

import anndata as ad
import pytest
import scanpy as sc
from spatialdata import SpatialData

import segtraq as st


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

    # adding raw counts etc.
    adata = sdata_new.tables["table"]
    adata.layers["raw"] = adata.X.copy()
    # normalizing and log-transforming the counts
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    # computing a PCA and neighbors
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)

    # this is important, because the test object initially contains some duplicate nucleus_ids
    # by calling validate_spatialdata,
    # we ensure that these get resolved before continuting with the tests
    st.validate_spatialdata(sdata_new, images_key="image", tables_centroid_x_key=None, tables_centroid_y_key=None)

    return sdata_new


@pytest.fixture(scope="session", name="segtraq_obj")
def test_segtraq_obj():
    """Load the SpatialData test sample once per test session."""

    test_data_path = Path(__file__).parent / "data" / "xenium.zarr"
    sdata_new = SpatialData.read(test_data_path)

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

    # adding raw counts etc.
    adata = sdata_new.tables["table"]
    adata.layers["raw"] = adata.X.copy()
    # normalizing and log-transforming the counts
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    # computing a PCA and neighbors
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)

    # creating a segtraq object
    return st.SegTraQ(
        sdata_new, tables_centroid_x_key="x_centroid", tables_centroid_y_key="y_centroid", images_key="image"
    )


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
        ref_cell_type="celltype_major",
        query_ensemble_key=None,
        inplace=True,
    )
    return sdata_new
