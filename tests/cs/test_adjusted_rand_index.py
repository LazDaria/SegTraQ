import numpy as np
import pytest

import segtraq as st


def test_adjusted_rand_index(sdata_new):
    ari = st.cs.adjusted_rand_index(sdata_new, resolution=1.0, key_prefix="leiden_subset")
    assert isinstance(ari, float), "ARI should be a float"
    assert -1 <= ari <= 1, "ARI should be in the range [-1, 1]"
    assert "mean_ari" in sdata_new.tables["table"].uns.keys(), "'mean_ari' should be present in uns"


def test_adjusted_rand_index_invalid_frac_cells_subset(sdata_new):
    with pytest.raises(ValueError):
        st.cs.adjusted_rand_index(
            sdata_new,
            resolution=1.0,
            key_prefix="leiden_subset",
            frac_cells_subset=1.5,  # Invalid fraction > 1
        )


def test_adjusted_rand_index_invalid_leiden_kwargs(sdata_new):
    with pytest.raises(TypeError):
        st.cs.adjusted_rand_index(
            sdata_new, resolution=1.0, key_prefix="leiden_subset", leiden_kwargs={"invalid_key": "invalid_value"}
        )


def test_adjusted_rand_index_single_cluster(sdata_new):
    # Use a very low resolution to force a single cluster
    ari = st.cs.adjusted_rand_index(sdata_new, resolution=0, key_prefix="leiden_single_cluster")
    assert np.isnan(ari), "ARI should be NaN when only one cluster is produced"
    assert "mean_ari" in sdata_new.tables["table"].uns.keys(), "'mean_ari' should be present in uns"
