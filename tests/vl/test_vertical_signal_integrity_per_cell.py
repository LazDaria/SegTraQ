import ovrlpy
import pandas as pd

import segtraq as st


def run_ovrlpy(sdata, n_comp, points_gene_key="feature_name", points_cell_id_key="cell_id", n_workers=8):
    coordinate_df = sdata.points["transcripts"].rename(columns={points_gene_key: "gene"})
    coordinate_df = coordinate_df.loc[:, ["gene", "x", "y", "z", points_cell_id_key]].compute()
    coordinate_df["z"] = coordinate_df["z"] - coordinate_df["z"].min()
    ovrlpy_sdata = ovrlpy.Ovrlp(coordinate_df, n_components=n_comp, n_workers=n_workers, random_state=42)
    ovrlpy_sdata.analyse()

    return ovrlpy_sdata


def test_vertical_signal_integrity_per_cell_type(sdata_new):
    n_celltypes = 10
    ovrlp = run_ovrlpy(sdata_new, n_comp=n_celltypes)

    df = st.vl.vertical_signal_integrity_per_cell(sdata_new, ovrlp)

    assert isinstance(df, pd.DataFrame), f"compute_sim_top_bottom_z should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "vertical_signal_integrity"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {df.columns}"


def test_vertical_signal_integrity_per_cell_values(sdata_new):
    n_celltypes = 10
    ovrlp = run_ovrlpy(sdata_new, n_comp=n_celltypes)

    df = st.vl.vertical_signal_integrity_per_cell(sdata_new, ovrlp)

    assert df["vertical_signal_integrity"].dropna().between(-1, 1).all()


def test_vertical_signal_integrity_inplace_writes_to_obs(sdata_new):
    n_celltypes = 5
    ovrlp = run_ovrlpy(sdata_new, n_comp=n_celltypes)

    key = "vertical_signal_integrity"
    obs = sdata_new.tables["table"].obs
    if key in obs.columns:
        sdata_new.tables["table"].obs = obs.drop(columns=[key])

    st.vl.vertical_signal_integrity_per_cell(sdata_new, ovrlp, inplace=True)
    assert key in sdata_new.tables["table"].obs.columns
