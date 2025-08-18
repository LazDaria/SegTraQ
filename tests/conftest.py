from pathlib import Path

import pytest
from spatialdata import SpatialData


@pytest.fixture(scope="session", name="sdata_new")
def test_sdata_new():
    """Load the SpatialData test sample once per test session."""

    test_data_path = Path(__file__).parent / "data" / "xenium_sp_subset.zarr"
    sdata_new = SpatialData.read(test_data_path)
    return sdata_new
