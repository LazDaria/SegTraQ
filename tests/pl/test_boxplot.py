import pandas as pd

import segtraq


def test_boxplot(segtraq_obj):
    st_dict = {"test1": segtraq_obj, "test2": segtraq_obj}
    boxplot_results = segtraq.pl.boxplot(st_dict, celltype_col="transferred_cell_type", value_key="transcript_count")
    # check that the output is a DataFrame with the expected columns
    assert isinstance(boxplot_results, pd.DataFrame)
    assert set(boxplot_results.columns) == {"Segmentation Method", "Cell Type", "value", "variable"}


def test_boxplot_combined(segtraq_obj):
    st_dict = {"test1": segtraq_obj, "test2": segtraq_obj}
    boxplot_results = segtraq.pl.boxplot_combined(
        st_dict, celltype_col="transferred_cell_type", value_key="transcript_count"
    )
    # check that the output is a DataFrame with the expected columns
    assert isinstance(boxplot_results, pd.DataFrame)
    assert set(boxplot_results.columns) == {"Segmentation Method", "Cell Type", "value", "variable"}
