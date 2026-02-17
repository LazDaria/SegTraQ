import segtraq as st


def test_silhouette_score(sdata_new):
    silhouette_score = st.cs.silhouette_score(sdata_new, resolution=1.0, key_prefix="leiden_subset", random_state=42)
    assert isinstance(silhouette_score, float), "Silhouette score should be a float"
    assert -1 <= silhouette_score <= 1, "Silhouette score should be in the range [-1, 1]"
    assert "silhouette_score" in sdata_new.tables["table"].uns.keys(), (
        "Silhouette score should be stored in sdata_new.uns"
    )
