import segtraq as st


def test_compute_purity(sdata):
    purity = st.cs.compute_purity(sdata, resolution=1.0, n_genes_subset=100, key_prefix="leiden_subset")
    assert isinstance(purity, float), "Purity should be a float"
    assert 0 <= purity <= 1, "Purity should be in the range [0, 1]"
