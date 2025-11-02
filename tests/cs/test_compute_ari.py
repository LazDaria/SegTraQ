import segtraq as st


def test_compute_ari(sdata_new):
    ari = st.cs.compute_ari(sdata_new, resolution=1.0, n_genes_subset=100, key_prefix="leiden_subset")
    assert isinstance(ari, float), "ARI should be a float"
    assert -1 <= ari <= 1, "ARI should be in the range [-1, 1]"
    assert "mean_ari" in sdata_new.tables["table"].uns.keys(), "'mean_ari' should be present in uns"
