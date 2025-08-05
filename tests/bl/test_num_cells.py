import pytest

import segtraq as st


def test_num_cells(sdata):
    num_cells = st.bl.num_cells(sdata)
    assert num_cells == 153, f"Expected 153 cells, found {num_cells}"

    num_nuclei = st.bl.num_cells(sdata, shape_key="nucleus_boundaries")
    assert num_nuclei == 129, f"Expected 129 cells, found {num_nuclei}"


def test_num_cells_invalid_key(sdata):
    with pytest.raises(KeyError):
        st.bl.num_cells(sdata, shape_key="invalid_key")
