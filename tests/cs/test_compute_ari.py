import segtraq as st


def test_compute_ari(sdata):
    ari = st.cs.compute_ari(sdata, resolution=1.0, n_genes_subset=100, key_prefix="leiden_subset")
    assert isinstance(ari, float), "ARI should be a float"
    assert -1 <= ari <= 1, "ARI should be in the range [-1, 1]"
