import numpy as np
import pytest

import segtraq as st


def test_cluster_connectedness(sdata_new):
    cc = st.cs.cluster_connectedness(sdata_new, resolution=1.0, key_prefix="leiden_subset", random_state=42)
    assert isinstance(cc, float), "Cluster connectedness should be a float"
    assert cc >= 0, "Cluster connectedness should be non-negative"
    assert "cluster_connectedness" in sdata_new.tables["table"].uns.keys(), (
        "'cluster_connectedness' should be present in uns"
    )


def test_cluster_connectedness_invalid_resolution(sdata_new):
    with pytest.raises(ValueError):
        st.cs.cluster_connectedness(
            sdata_new,
            resolution=-1,
            key_prefix="leiden_subset",
            random_state=42,
        )


def test_cluster_connectedness_single_cluster(sdata_new):
    # Use a very low resolution to force a single cluster
    cc = st.cs.cluster_connectedness(sdata_new, resolution=0, key_prefix="leiden_single_cluster", random_state=42)
    assert np.isnan(cc), "Cluster connectedness should be NaN when only one cluster is produced"
    assert "cluster_connectedness" in sdata_new.tables["table"].uns.keys(), (
        "'cluster_connectedness' should be present in uns"
    )
