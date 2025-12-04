import pytest

import segtraq as st


def test_num_cells(sdata_new):
    num_cells = st.bl.num_cells(sdata_new)
    assert num_cells == 743, f"Expected 743 cells, found {num_cells}"

    num_nuclei = st.bl.num_cells(sdata_new, tables_key="table")
    assert num_nuclei == 743, f"Expected 743 nuclei, found {num_nuclei}"

    assert "num_cells" in sdata_new.tables["table"].uns.keys(), "'num_cells' should be present in uns"


def test_num_cells_invalid_key(sdata_new):
    with pytest.raises(KeyError):
        st.bl.num_cells(sdata_new, tables_key="invalid_key")
