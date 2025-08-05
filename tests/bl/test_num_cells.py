import segtraq as st
import pytest

def test_num_cells(sdata):
    num_cells = st.bl.num_cells(sdata)
    assert num_cells == 138, f"Expected 3 cells, found {num_cells}"
    
    num_nuclei = st.bl.num_cells(sdata, shape_key="nucleus_boundaries")
    assert num_nuclei == 123, f"Expected 3 cells, found {num_cells}"

def test_num_cells_invalid_key(sdata):
    with pytest.raises(KeyError):
        st.bl.num_cells(sdata, shape_key="invalid_key")