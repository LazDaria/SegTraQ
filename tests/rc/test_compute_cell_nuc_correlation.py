import pandas as pd
import pytest

import segtraq as st


def test_data_types_and_columns(sdata_new):
    # Prepare table.obs with best_nuc_id and IoU
    st.validate_spatialdata(sdata_new, images_key="image", tables_centroid_x_key=None, tables_centroid_y_key=None)
    corr_df = st.rc.compute_cell_nuc_correlation(sdata_new, n_jobs=8)
    assert isinstance(corr_df, pd.DataFrame), f"Expected DataFrame, got {type(corr_df)}"
    exp = {"cell_id", "best_nuc_id", "IoU", "nucleus_fraction", "corr_nc_cell"}
    assert set(corr_df.columns) == exp, f"Columns mismatch: expected {exp}, got {set(corr_df.columns)}"
    assert corr_df["corr_nc_cell"].dtype == float, (
        f"Expected correlation dtype float, got {corr_df['corr_nc_cell'].dtype}"
    )


def test_unsupported_metric_raises_value_error(sdata_new):
    with pytest.raises(ValueError, match="Metric dummy_metric not supported"):
        st.rc.compute_cell_nuc_correlation(sdata_new, metric="dummy_metric", n_jobs=8)
