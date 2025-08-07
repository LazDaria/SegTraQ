import pytest
from pathlib import Path
import os
import datetime
from spatialdata import SpatialData

@pytest.fixture(scope="session", name="sdata_new")
def test_sdata_new():
    """Load the SpatialData test sample once per test session."""
    test_data_path = Path(__file__).parent / "data" / "test_sample.zarr"

    # Print the full path
    print(f"Test data path: {test_data_path.resolve()}")

    # Print file creation and modification times
    try:
        stat = test_data_path.stat()
        ctime = datetime.datetime.fromtimestamp(stat.st_ctime)
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
        print(f"Created: {ctime}")
        print(f"Modified: {mtime}")
    except Exception as e:
        print(f"Error accessing file stats: {e}")

    # Intentionally raise an error to test behavior
    raise RuntimeError("Intentional error for debugging")

    # This part will not be reached due to the error above
    sdata = SpatialData.read(test_data_path)
    return sdata
