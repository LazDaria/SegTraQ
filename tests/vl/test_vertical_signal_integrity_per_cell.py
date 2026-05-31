import pandas as pd

import segtraq as st


def test_vertical_signal_integrity_per_cell_type(sdata_new):
    n_celltypes = 10

    df = st.vl.vertical_signal_integrity_per_cell(sdata_new, ovrlpy_init_kwargs={"n_components": n_celltypes})

    assert isinstance(df, pd.DataFrame), f"compute_sim_top_bottom_z should return a DataFrame, got {type(df)}"
    expected_cols = {"cell_id", "vertical_signal_integrity"}
    assert set(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {df.columns}"


def test_vertical_signal_integrity_per_cell_values(sdata_new):
    n_celltypes = 10

    df = st.vl.vertical_signal_integrity_per_cell(sdata_new, ovrlpy_init_kwargs={"n_components": n_celltypes})

    assert df["vertical_signal_integrity"].dropna().between(-1, 1).all()


def test_vertical_signal_integrity_inplace_writes_to_obs(sdata_new):
    n_celltypes = 5

    key = "vertical_signal_integrity"
    obs = sdata_new.tables["table"].obs
    if key in obs.columns:
        sdata_new.tables["table"].obs = obs.drop(columns=[key])

    st.vl.vertical_signal_integrity_per_cell(sdata_new, ovrlpy_init_kwargs={"n_components": n_celltypes}, inplace=True)
    assert key in sdata_new.tables["table"].obs.columns
