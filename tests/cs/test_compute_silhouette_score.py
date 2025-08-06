import segtraq as st


def test_compute_silhouette_score(sdata):
    silhouette_score = st.cs.compute_silhouette_score(
        sdata, resolution=1.0, key_prefix="leiden_subset", random_state=42
    )
    assert isinstance(silhouette_score, float), "Silhouette score should be a float"
    assert -1 <= silhouette_score <= 1, "Silhouette score should be in the range [-1, 1]"
