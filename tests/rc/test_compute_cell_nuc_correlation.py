
import pandas as pd
import pytest

import segtraq as st


def test_data_types_and_columns(sdata_new):
    # Prepare table.obs with best_nuc_id and IoU
    corr_df, iou_df = st.rc.compute_cell_nuc_correlation(sdata_new, n_jobs_iou=8)
    assert isinstance(corr_df, pd.DataFrame), f"Expected DataFrame, got {type(corr_df)}"
    assert isinstance(iou_df, pd.DataFrame), f"Expected DataFrame, got {type(iou_df)}"
    assert set(corr_df.columns) == {
        "cell_id",
        "best_nuc_id",
        "IoU",
        "corr_nc_cell",
    }, f"Columns mismatch: expected {{'cell_id','best_nuc_id','IoU','corr_nc_cell'}}, got {set(corr_df.columns)}"
    assert (
        corr_df["corr_nc_cell"].dtype == float
    ), f"Expected correlation dtype float, got {corr_df['corr_nc_cell'].dtype}"


def test_unsupported_metric_raises_value_error(sdata_new):
    with pytest.raises(ValueError, match="Metric dummy_metric not supported"):
        st.rc.compute_cell_nuc_correlation(sdata_new, metric="dummy_metric", n_jobs_iou=8)
