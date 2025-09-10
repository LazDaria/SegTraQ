import segtraq as st


def test_compute_mcd(sdata_new):
    mcd = st.cs.compute_mean_cosine_distance(sdata_new, resolution=1.0, key_prefix="leiden_subset", random_state=42)
    assert isinstance(mcd, float), "MCD should be a float"
    assert mcd >= 0, "MCD should be non-negative"
