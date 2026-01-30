import pandas as pd
import segtraq as st


def test_compute_sim_top_bottom_z_type(sdata_new):
    df = st.vl.compute_sim_top_bottom_z(sdata_new)

    assert isinstance(df, pd.DataFrame), f"compute_sim_top_bottom_z should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "cosine_sim_top_bottom_z"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {df.columns}"
    
def test_compute_sim_top_bottom_z_values(sdata_new):
    df = st.vl.compute_sim_top_bottom_z(sdata_new)
    
    assert df['cosine_sim_top_bottom_z'].dropna().between(-1, 1).all()