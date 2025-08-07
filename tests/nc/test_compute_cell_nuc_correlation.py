import pandas as pd
import pytest

import segtraq as st


# def test_data_types_and_columns(sdata_new):
#     # Prepare table.obs with best_nuc_id and IoU
#     corr_df = st.nc.compute_cell_nuc_correlation(sdata_new)
#     assert isinstance(corr_df, pd.DataFrame), f"Expected DataFrame, got {type(corr_df)}"
#     assert set(corr_df.columns) == {
#         "cell_id",
#         "best_nuc_id",
#         "IoU",
#         "correlation",
#     }, f"Columns mismatch: expected {{'cell_id','best_nuc_id','IoU','correlation'}}, got {set(corr_df.columns)}"
#     assert corr_df["correlation"].dtype == float, (
#         f"Expected correlation dtype float, got {corr_df['correlation'].dtype}"
#     )


# def test_unsupported_metric_raises_value_error(sdata_new):
#     with pytest.raises(ValueError, match="Metric spearman not supported"):
#         st.nc.compute_cell_nuc_correlation(sdata_new, metric="spearman")
