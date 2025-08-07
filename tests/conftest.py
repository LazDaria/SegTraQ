from pathlib import Path

import pytest
from spatialdata import SpatialData


@pytest.fixture(scope="session", name="sdata")
def test_sdata():
    """Load the SpatialData test sample once per test session."""
    test_data_path = Path(__file__).parent / "data" / "test_sample_new.zarr"
    sdata = SpatialData.read(test_data_path)
    return sdata
