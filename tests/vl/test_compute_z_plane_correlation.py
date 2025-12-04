import pandas as pd

import segtraq as st


def test_compute_z_plane_correlation(sdata_new):
    z_plane_correlation = st.vl.compute_z_plane_correlation(sdata_new)
    assert isinstance(z_plane_correlation, pd.DataFrame), "Z-plane correlation should be a DataFrame"
    assert "correlation" in z_plane_correlation.columns, "DataFrame should have a 'correlation' column"
    assert all(z_plane_correlation["correlation"].between(-1, 1)), "Correlations should be between -1 and 1"
    assert "correlation" in sdata_new.tables["table"].obs.columns, (
        "Correlation values should be added to sdata_new tables' obs"
    )
