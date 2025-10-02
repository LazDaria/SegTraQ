import segtraq as st


def test_read_xenium(xenium_directory):
    sdata = st.io.read_xenium(xenium_directory)
    assert "nucleus_labels" in sdata.labels, "Nucleus labels should be present in the SpatialData object"
    assert "cell_labels" in sdata.labels, "Cell labels should be present in the SpatialData object"
    assert len(sdata.points["transcripts"]) > 0, "There should be some transcript points"
    assert sdata.tables["table"].n_vars > 0, "There should be some genes in the AnnData object"
    assert sdata.tables["table"].n_obs > 0, "There should be some cells in the AnnData object"
