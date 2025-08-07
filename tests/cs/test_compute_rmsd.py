import segtraq as st


def test_compute_rmsd(sdata):
    rmsd = st.cs.compute_rmsd(sdata, resolution=1.0, key_prefix="leiden_subset", random_state=42)
    assert isinstance(rmsd, float), "RMSD should be a float"
    assert rmsd >= 0, "RMSD should be non-negative"
