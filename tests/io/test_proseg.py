import segtraq as st


def test_read_proseg_v2(proseg_v2_directory, xenium_directory):
    sdata = st.io.read_proseg_2(proseg_v2_directory, xenium_directory)
    assert "cell_labels" in sdata.labels, "Cell labels should be present in the SpatialData object"
    assert len(sdata.points["transcripts"]) > 0, "There should be some transcript points"
    assert sdata.tables["table"].n_vars > 0, "There should be some genes in the AnnData object"
    assert sdata.tables["table"].n_obs > 0, "There should be some cells in the AnnData object"


def test_read_proseg_v3(proseg_v3_directory, xenium_directory):
    sdata = st.io.read_proseg_3(proseg_v3_directory, xenium_directory)
    assert "cell_labels" in sdata.labels, "Cell labels should be present in the SpatialData object"
    assert len(sdata.points["transcripts"]) > 0, "There should be some transcript points"
    assert sdata.tables["table"].n_vars > 0, "There should be some genes in the AnnData object"
    assert sdata.tables["table"].n_obs > 0, "There should be some cells in the AnnData object"
