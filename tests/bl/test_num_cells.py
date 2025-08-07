import pytest

import segtraq as st


def test_num_cells(sdata):
    num_cells = st.bl.num_cells(sdata)
    assert num_cells == 158, f"Expected 158 cells, found {num_cells}"

    num_nuclei = st.bl.num_cells(sdata, shape_key="nucleus_boundaries")
    assert num_nuclei == 132, f"Expected 132 cells, found {num_nuclei}"


def test_num_cells_invalid_key(sdata):
    with pytest.raises(KeyError):
        st.bl.num_cells(sdata, shape_key="invalid_key")
