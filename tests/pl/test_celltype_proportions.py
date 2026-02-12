import numpy as np
import pandas as pd

import segtraq


def test_celltype_proportions(segtraq_obj):
    st_dict = {"test1": segtraq_obj, "test2": segtraq_obj}
    ct_proportions = segtraq.pl.celltype_proportions(st_dict, celltype_col="transferred_cell_type")
    # check that the output is a DataFrame with the expected columns
    assert isinstance(ct_proportions, pd.DataFrame)
    assert set(ct_proportions.columns) == {"Segmentation Method", "Cell Type", "Count"}
    # check that both datasets have the same total count of cells
    np.all(ct_proportions.groupby("Segmentation Method")["Count"].sum() == 751)
