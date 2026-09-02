import numpy as np
import pytest

import segtraq as st


def test_silhouette_score(sdata_new):
    silhouette_score = st.cs.silhouette_score(sdata_new, resolution=1.0, key_prefix="leiden_subset", random_state=42)
    assert isinstance(silhouette_score, float), "Silhouette score should be a float"
    assert -1 <= silhouette_score <= 1, "Silhouette score should be in the range [-1, 1]"
    assert "silhouette_score" in sdata_new.tables["table"].uns.keys(), (
        "Silhouette score should be stored in sdata_new.uns"
    )


def test_silhouette_score_invalid_resolution(sdata_new):
    with pytest.raises(ValueError):
        st.cs.silhouette_score(
            sdata_new,
            resolution=-0.5,
            key_prefix="leiden_subset",
            random_state=42,  # Invalid negative resolution
        )


def test_silhouette_score_single_cluster(sdata_new):
    # Use a very low resolution to force a single cluster
    silhouette_score = st.cs.silhouette_score(
        sdata_new, resolution=0.0001, key_prefix="leiden_single_cluster", random_state=42
    )
    assert np.isnan(silhouette_score), "Silhouette score should be NaN when only one cluster is produced"
    assert "silhouette_score" in sdata_new.tables["table"].uns.keys(), (
        "Silhouette score should be stored in sdata_new.uns"
    )
