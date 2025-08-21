import segtraq as st


def test_create_spatialdata(
    bidcell_image,
    bidcell_cell_labels,
    bidcell_nucleus_labels,
    bidcell_transcripts,
):
    sdata = st.fs.create_spatialdata(
        points=bidcell_transcripts,
        labels={"cell_labels": bidcell_cell_labels, "nucleus_labels": bidcell_nucleus_labels},
        images=bidcell_image,
        # optional, if your coordinates are not named x, y, z
        coord_columns=["x_location", "y_location", "z_location"],
    )

    # testing if compute_shapes works
    sdata = st.fs.compute_shapes(sdata, labels_key="cell_labels")
    assert "transcripts" in sdata.points, "Expected 'transcripts' to be in the SpatialData object"
