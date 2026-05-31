# import segtraq as st


# def test_cluster_connectedness(sdata_new):
#     cc = st.cs.cluster_connectedness(sdata_new, resolution=1.0, key_prefix="leiden_subset", random_state=42)
#     assert isinstance(cc, float), "Cluster connectedness should be a float"
#     assert cc >= 0, "Cluster connectedness should be non-negative"
#     assert "cluster_connectedness" in sdata_new.tables["table"].uns.keys(), (
#         "'cluster_connectedness' should be present in uns"
#     )
