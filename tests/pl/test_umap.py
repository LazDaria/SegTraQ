import pandas as pd

import segtraq


def test_umap(segtraq_obj):
    st_dict = {"test1": segtraq_obj, "test2": segtraq_obj}
    umap_results = segtraq.pl.umap(st_dict, color="transferred_cell_type")
    # check that the output is a DataFrame with the expected columns
    assert isinstance(umap_results, pd.DataFrame)
    assert set(umap_results.columns) == {"x", "y", "Segmentation Method", "transferred_cell_type"}
