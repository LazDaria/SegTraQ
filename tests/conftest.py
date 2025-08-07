import pytest
from spatialdata import SpatialData


@pytest.fixture(scope="session", name="sdata")
def test_sdata():
    """Load the SpatialData test sample once per test session."""
    # test_data_path = Path(__file__).parent / "data" / "test_sample.zarr"
    test_data_path = "/g/huber/projects/CODEX/segtraq/data/xenium_sp_subset.zarr"
    sdata = SpatialData.read(test_data_path)
    return sdata
