import numpy as np
import pandas as pd
import pytest
import spatialdata as sd

import segtraq as st


def _pick_two_cells_with_different_types(obs: pd.DataFrame, cell_key="cell", cell_type_key="transferred_cell_type"):
    """Return two cells with different non-null types."""
    sub = obs[[cell_key, cell_type_key]].dropna()
    if sub.empty:
        raise RuntimeError("No non-null transferred_cell_type labels found in obs.")

    # Find two different types
    types = sub[cell_type_key].astype("object")
    if types.nunique() < 2:
        raise RuntimeError("Need at least two distinct cell types to build heterotypic test case.")

    # pick one cell from first type and one from second type
    t0, t1 = list(types.unique())[:2]
    cid0 = sub.loc[types == t0, cell_key].iloc[0]
    cid1 = sub.loc[types == t1, cell_key].iloc[0]
    return cid0, cid1


def test_fraction_heterotypic_overlap_runs_and_returns_expected_columns(sdata_3D_labeled):
    shapes_key_list = ["cell_boundaries_z0", "cell_boundaries_z1", "cell_boundaries_z2", "cell_boundaries_z3"]
    df = st.vl.fraction_heterotypic_overlap(
        sdata_3D_labeled,
        tables_cell_id_key="cell",
        shapes_cell_id_key="cell",
        shapes_key_list=shapes_key_list,
        inplace=False,
    )
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"cell", "heterotypic_overlap_area", "heterotypic_overlap_fraction"}


def test_fraction_heterotypic_overlap_inplace_writes_to_obs(sdata_3D_labeled):
    sdata = sd.deepcopy(sdata_3D_labeled)

    # drop existing cols if present (avoid false positives if tests re-run)
    obs = sdata.tables["table"].obs
    for col in ["heterotypic_overlap_area", "heterotypic_overlap_fraction"]:
        if col in obs.columns:
            obs = obs.drop(columns=[col])
    sdata.tables["table"].obs = obs

    shapes_key_list = ["cell_boundaries_z0", "cell_boundaries_z1", "cell_boundaries_z2", "cell_boundaries_z3"]
    st.vl.fraction_heterotypic_overlap(
        sdata, tables_cell_id_key="cell", shapes_cell_id_key="cell", shapes_key_list=shapes_key_list, inplace=True
    )

    obs2 = sdata.tables["table"].obs
    assert "heterotypic_overlap_area" in obs2.columns
    assert "heterotypic_overlap_fraction" in obs2.columns


def test_fraction_heterotypic_overlap_forced_full_overlap_two_cells(sdata_3D_labeled):
    """
    Build two z-layers from real polygons and real cell IDs.

    We pick two cells with different transferred_cell_type labels: cid_a, cid_b.
    Then we create:
      - z0 layer: polygon of cid_a
      - z1 layer: cell = cid_b but geometry overwritten to equal cid_a's polygon

    Because geometries are identical and types differ, each should get overlap_fraction ~ 1.
    """
    sdata = sd.deepcopy(sdata_3D_labeled)

    shapes = sdata.shapes["cell_boundaries"].copy()

    obs = sdata.tables["table"].obs
    cid_a, cid_b = _pick_two_cells_with_different_types(obs)

    # Ensure both cids exist in shapes
    if cid_a not in shapes.index or cid_b not in shapes.index:
        pytest.skip("Chosen cell IDs not found in shapes index; cannot construct forced overlap case.")

    z0 = shapes.loc[[cid_a]].copy()
    z1 = shapes.loc[[cid_b]].copy()

    # Force full overlap by copying geometry from cid_a onto cid_b in the other z-layer
    z1["geometry"] = [z0.geometry.iloc[0]]

    # Install these layers under the expected keys
    sdata.shapes["cell_boundaries_z0"] = z0
    sdata.shapes["cell_boundaries_z1"] = z1

    out = st.vl.fraction_heterotypic_overlap(
        sdata,
        tables_cell_id_key="cell",
        shapes_cell_id_key="cell",
        shapes_key_list=["cell_boundaries_z0", "cell_boundaries_z1"],
        cell_type_key="transferred_cell_type",
        inplace=False,
        unknown_policy="treat_as_label",
    ).set_index("cell")

    # Both cells should have ~100% heterotypic overlap
    assert np.isfinite(out.loc[cid_a, "heterotypic_overlap_fraction"])
    assert np.isfinite(out.loc[cid_b, "heterotypic_overlap_fraction"])
    assert np.isclose(out.loc[cid_a, "heterotypic_overlap_fraction"], 1.0, atol=1e-6)
    assert np.isclose(out.loc[cid_b, "heterotypic_overlap_fraction"], 1.0, atol=1e-6)

    # Overlap area should be positive and <= polygon area (implicitly)
    assert out.loc[cid_a, "heterotypic_overlap_area"] > 0
    assert out.loc[cid_b, "heterotypic_overlap_area"] > 0


def test_fraction_heterotypic_overlap_unknown_exclude_returns_nan_for_focal(sdata_3D_labeled):
    """
    Hit the unknown_policy='exclude' branch:
    If the focal cell has unknown type, its overlap fraction should be NaN.
    """
    sdata = sd.deepcopy(sdata_3D_labeled)

    shapes = sdata.shapes["cell_boundaries"].copy()

    obs = sdata.tables["table"].obs
    cid_a, cid_b = _pick_two_cells_with_different_types(obs)

    if cid_a not in shapes.index or cid_b not in shapes.index:
        pytest.skip("Chosen cell IDs not found in shapes index; cannot construct forced overlap case.")

    # Make cid_a unknown in the table
    obs2 = obs.copy()
    obs2.loc[obs2["cell"] == cid_a, "transferred_cell_type"] = np.nan
    sdata.tables["table"].obs = obs2

    # Build z-layers (same forced-overlap construction)
    z0 = shapes.loc[[cid_a]].copy()
    z1 = shapes.loc[[cid_b]].copy()
    z1["geometry"] = [z0.geometry.iloc[0]]
    sdata.shapes["cell_boundaries_z0"] = z0
    sdata.shapes["cell_boundaries_z1"] = z1

    out = st.vl.fraction_heterotypic_overlap(
        sdata,
        tables_cell_id_key="cell",
        shapes_cell_id_key="cell",
        shapes_key_list=["cell_boundaries_z0", "cell_boundaries_z1"],
        cell_type_key="transferred_cell_type",
        inplace=False,
        unknown_policy="exclude",
    ).set_index("cell")

    assert np.isnan(out.loc[cid_a, "heterotypic_overlap_fraction"])
