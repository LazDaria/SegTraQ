import pandas as pd

import segtraq as st


def test_compute_cell_cell_IoU(sdata_new):
    cell_cell_IoU = st.vl.compute_cell_cell_IoU(sdata_new)
    assert isinstance(cell_cell_IoU, pd.DataFrame), "Z-plane correlation should be a DataFrame"
    assert "IoU" in cell_cell_IoU.columns, "DataFrame should have a 'IoU' column"
    assert "IoU_sum" in cell_cell_IoU.columns, "DataFrame should have a 'IoU_sum' column"
    assert all(cell_cell_IoU["IoU_sum"]>=0), "IoU_sum should be positive"
    assert "IoU_sum" in sdata_new.tables["table"].obs.columns, (
        "IoU_sum values should be added to sdata_new tables' obs"
    )
