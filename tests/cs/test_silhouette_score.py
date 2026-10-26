import numpy as np
import pandas as pd
import pytest

import segtraq as st


def test_silhouette_score(sdata_new):
    silhouette_score = st.cs.silhouette_score(sdata_new, resolution=1.0, key_prefix="leiden_subset", random_state=42)
    assert isinstance(silhouette_score, float), "Silhouette score should be a float"
    assert -1 <= silhouette_score <= 1, "Silhouette score should be in the range [-1, 1]"
    assert "silhouette_score_leiden" in sdata_new.tables["table"].uns.keys(), (
        "Silhouette score should be stored in sdata_new.uns"
    )


def test_silhouette_score_invalid_resolution(sdata_new):
    with pytest.raises(ValueError):
        st.cs.silhouette_score(
            sdata_new,
            resolution=-1,
            key_prefix="leiden_subset",
            random_state=42,  # Invalid negative resolution
        )


def test_silhouette_score_single_cluster(sdata_new):
    # Use a very low resolution to force a single cluster
    silhouette_score = st.cs.silhouette_score(
        sdata_new, resolution=0, key_prefix="leiden_single_cluster", random_state=42
    )
    assert np.isnan(silhouette_score), "Silhouette score should be NaN when only one cluster is produced"
    assert "silhouette_score_leiden" in sdata_new.tables["table"].uns.keys(), (
        "Silhouette score should be stored in sdata_new.uns"
    )


def test_silhouette_score_cell_type_key(sdata_new):
    cell_type_key = "test_cell_type"
    adata = sdata_new.tables["table"]
    # Simple synthetic labels with two groups
    adata.obs[cell_type_key] = pd.Categorical(np.where(np.arange(adata.n_obs) % 2 == 0, "A", "B"))
    adata.uns.pop("silhouette_score_leiden", None)

    silhouette_score = st.cs.silhouette_score(sdata_new, cell_type_key=cell_type_key)

    # merge_into_uns may replace the table object, so re-fetch it
    uns = sdata_new.tables["table"].uns
    assert isinstance(silhouette_score, float), "Silhouette score should be a float"
    assert -1 <= silhouette_score <= 1, "Silhouette score should be in the range [-1, 1]"
    assert f"silhouette_score_{cell_type_key}" in uns, f"'silhouette_score_{cell_type_key}' should be present in uns"
    assert uns[f"silhouette_score_{cell_type_key}"] == pytest.approx(silhouette_score)
    assert "silhouette_score_leiden" not in uns, (
        "'silhouette_score_leiden' should not be written when a cell_type_key is provided"
    )


def test_silhouette_score_cell_type_key_with_nan(sdata_new):
    adata = sdata_new.tables["table"]
    cell_type_key = "test_cell_type_nan"
    labels = np.where(np.arange(adata.n_obs) % 2 == 0, "A", "B").astype(object)
    labels[::5] = np.nan  # some unlabelled cells
    adata.obs[cell_type_key] = pd.Categorical(labels)

    silhouette_score = st.cs.silhouette_score(sdata_new, cell_type_key=cell_type_key)

    assert isinstance(silhouette_score, float), "Silhouette score should be a float"
    assert not np.isnan(silhouette_score), "NaN labels should be ignored, not propagate"
    assert -1 <= silhouette_score <= 1, "Silhouette score should be in the range [-1, 1]"


def test_silhouette_score_cell_type_key_missing(sdata_new):
    with pytest.raises(ValueError, match="not found"):
        st.cs.silhouette_score(sdata_new, cell_type_key="does_not_exist")


def test_silhouette_score_cell_type_key_single_label(sdata_new):
    adata = sdata_new.tables["table"]
    adata.obs["single_cell_type"] = pd.Categorical(["A"] * adata.n_obs)

    with pytest.raises(ValueError, match="more than one cluster"):
        st.cs.silhouette_score(sdata_new, cell_type_key="single_cell_type")
