import pandas as pd
import ovrlpy
import segtraq as st

def run_ovrlpy(sdata, n_comp, points_gene_key="feature_name", n_workers=8):
    coordinate_df = sdata["transcripts"].compute()
    coordinate_df = coordinate_df.rename(columns={points_gene_key: "gene"})
    coordinate_df = coordinate_df[["gene", "x", "y", "z"]]
    coordinate_df["z"] = coordinate_df["z"] - coordinate_df["z"].min()
    ovrlpy_sdata = ovrlpy.Ovrlp(coordinate_df, n_components=n_comp, n_workers=n_workers, random_state=42)
    ovrlpy_sdata.analyse()

    vsi_map = ovrlpy_sdata.integrity_map

    return vsi_map

def test_compute_mean_vsi_per_cell_type(sdata_new):
    n_celltypes = 10
    vsi_map = run_ovrlpy(sdata_new, n_comp=n_celltypes)
    
    df = st.vl.compute_mean_vsi_per_cell(sdata_new, vsi_map)

    assert isinstance(df, pd.DataFrame), f"compute_sim_top_bottom_z should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "mean_vsi"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {df.columns}"
    
def test_compute_mean_vsi_per_cell_values(sdata_new):
    n_celltypes = 10
    vsi_map = run_ovrlpy(sdata_new, n_comp=n_celltypes)
    
    df = st.vl.compute_mean_vsi_per_cell(sdata_new, vsi_map)
    
    assert df['mean_vsi'].dropna().between(-1, 1).all()