from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import tifffile
from spatialdata import SpatialData


@pytest.fixture(scope="session", name="sdata_new")
def test_sdata_new():
    """Load the SpatialData test sample once per test session."""

    test_data_path = Path(__file__).parent / "data" / "xenium_sp_subset.zarr"
    sdata_new = SpatialData.read(test_data_path)
    return sdata_new


@pytest.fixture(scope="session", name="xenium_cell_boundaries")
def test_xenium_cell_boundaries():
    path = Path(__file__).parent / "data/xenium/cell_boundaries.parquet"
    return pd.read_parquet(path)


@pytest.fixture(scope="session", name="xenium_nucleus_boundaries")
def test_xenium_nucleus_boundaries():
    path = Path(__file__).parent / "data/xenium/nucleus_boundaries.parquet"
    return pd.read_parquet(path)


@pytest.fixture(scope="session", name="xenium_cell_labels")
def test_xenium_cell_labels():
    path = Path(__file__).parent / "data/xenium/cell_mask_um.tif"
    return tifffile.imread(path)


@pytest.fixture(scope="session", name="xenium_nucleus_labels")
def test_xenium_nucleus_labels():
    path = Path(__file__).parent / "data/xenium/nuc_mask_um.tif"
    return tifffile.imread(path)


@pytest.fixture(scope="session", name="xenium_image")
def test_xenium_image():
    path = Path(__file__).parent / "data/xenium/dapi_um.tif"
    return tifffile.imread(path)


@pytest.fixture(scope="session", name="xenium_transcripts")
def test_xenium_transcripts():
    path = Path(__file__).parent / "data/xenium/transcripts.csv"
    return pd.read_csv(path)


@pytest.fixture(scope="session", name="proseg_cell_metadata")
def test_proseg_cell_metadata():
    path = Path(__file__).parent / "data/proseg/cell-metadata.csv.gz"
    return pd.read_csv(path, compression="gzip")


@pytest.fixture(scope="session", name="proseg_cell_boundaries")
def test_proseg_cell_boundaries():
    path = Path(__file__).parent / "data/proseg/cell-polygons-layers.geojson"
    return gpd.read_file(path)


@pytest.fixture(scope="session", name="bidcell_cell_labels")
def test_bidcell_cell_labels():
    path = Path(__file__).parent / "data/bidcell/cell_labels.tif"
    return tifffile.imread(path)


@pytest.fixture(scope="session", name="bidcell_image")
def test_bidcell_image():
    path = Path(__file__).parent / "data/bidcell/dapi_resized.tif"
    return tifffile.imread(path)


@pytest.fixture(scope="session", name="bidcell_nucleus_labels")
def test_bidcell_nucleus_labels():
    path = Path(__file__).parent / "data/bidcell/nucleus_labels.tif"
    return tifffile.imread(path)


@pytest.fixture(scope="session", name="bidcell_transcripts")
def test_bidcell_transcripts():
    path = Path(__file__).parent / "data/bidcell/transcripts_processed.csv"
    return pd.read_csv(path)
