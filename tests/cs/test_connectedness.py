import numpy as np
import pandas as pd
import pytest

import segtraq as st


def test_connectedness(sdata_new):
    cc = st.cs.connectedness(sdata_new, resolution=1.0, key_prefix="leiden_subset", random_state=42)
    assert isinstance(cc, float), "Cluster connectedness should be a float"
    assert cc >= 0, "Cluster connectedness should be non-negative"
    assert "connectedness_leiden" in sdata_new.tables["table"].uns.keys(), (
        "'connectedness_leiden' should be present in uns"
    )


def test_connectedness_invalid_resolution(sdata_new):
    with pytest.raises(ValueError):
        st.cs.connectedness(
            sdata_new,
            resolution=-1,
            key_prefix="leiden_subset",
            random_state=42,
        )


def test_connectedness_single_cluster(sdata_new):
    # Use a very low resolution to force a single cluster
    cc = st.cs.connectedness(sdata_new, resolution=0, key_prefix="leiden_single_cluster", random_state=42)
    assert np.isnan(cc), "Cluster connectedness should be NaN when only one cluster is produced"
    assert "connectedness_leiden" in sdata_new.tables["table"].uns.keys(), (
        "'connectedness_leiden' should be present in uns"
    )


def test_connectedness_cell_type_key(sdata_new):
    adata = sdata_new.tables["table"]
    cell_type_key = "test_cell_type"
    # Simple synthetic labels with two groups
    adata.obs[cell_type_key] = pd.Categorical(np.where(np.arange(adata.n_obs) % 2 == 0, "A", "B"))
    adata.uns.pop("connectedness_leiden", None)

    cc = st.cs.connectedness(sdata_new, cell_type_key=cell_type_key)

    # merge_into_uns may replace the table object, so re-fetch it
    uns = sdata_new.tables["table"].uns
    assert isinstance(cc, float), "Cluster connectedness should be a float"
    assert cc >= 0, "Cluster connectedness should be non-negative"
    assert f"connectedness_{cell_type_key}" in uns, f"'connectedness_{cell_type_key}' should be present in uns"
    assert uns[f"connectedness_{cell_type_key}"] == pytest.approx(cc)
    assert "connectedness_leiden" not in uns, (
        "'connectedness_leiden' should not be written when a cell_type_key is provided"
    )


def test_connectedness_cell_type_key_missing(sdata_new):
    with pytest.raises(ValueError, match="not found"):
        st.cs.connectedness(sdata_new, cell_type_key="does_not_exist")


def test_connectedness_cell_type_key_single_label(sdata_new):
    adata = sdata_new.tables["table"]
    adata.obs["single_cell_type"] = pd.Categorical(["A"] * adata.n_obs)

    with pytest.raises(ValueError, match="more than one cluster"):
        st.cs.connectedness(sdata_new, cell_type_key="single_cell_type")
