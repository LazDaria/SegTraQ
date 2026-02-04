import numpy as np
import ovrlpy
import pandas as pd
import pytest

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


def test_vertical_signal_integrity_per_cell_type(sdata_new):
    n_celltypes = 10
    vsi_map = run_ovrlpy(sdata_new, n_comp=n_celltypes)

    df = st.vl.vertical_signal_integrity_per_cell(sdata_new, vsi_map)

    assert isinstance(df, pd.DataFrame), f"compute_sim_top_bottom_z should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "vertical_signal_integrity"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {df.columns}"


def test_vertical_signal_integrity_per_cell_values(sdata_new):
    n_celltypes = 10
    vsi_map = run_ovrlpy(sdata_new, n_comp=n_celltypes)

    df = st.vl.vertical_signal_integrity_per_cell(sdata_new, vsi_map)

    assert df["vertical_signal_integrity"].dropna().between(-1, 1).all()


def test_vertical_signal_integrity_vsi_map_must_be_2d(sdata_new):
    vsi_map = np.zeros((10, 10, 2))
    with pytest.raises(ValueError, match="must be a 2D array"):
        st.vl.vertical_signal_integrity_per_cell(sdata_new, vsi_map)


def test_vertical_signal_integrity_inplace_writes_to_obs(sdata_new):
    n_celltypes = 5
    vsi_map = run_ovrlpy(sdata_new, n_comp=n_celltypes)

    key = "vertical_signal_integrity"
    obs = sdata_new.tables["table"].obs
    if key in obs.columns:
        sdata_new.tables["table"].obs = obs.drop(columns=[key])

    st.vl.vertical_signal_integrity_per_cell(sdata_new, vsi_map, inplace=True)
    assert key in sdata_new.tables["table"].obs.columns
