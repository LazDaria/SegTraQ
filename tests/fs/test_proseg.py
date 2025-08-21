import segtraq as st


def test_create_spatialdata(
    xenium_image, xenium_transcripts, proseg_cell_metadata, proseg_cell_boundaries
):  # this dictionary maps from a cell ID (e. g. bfnbkogm-1) to a numeric ID (e. g. 1)
    cell_id_dict = dict(zip(proseg_cell_metadata["original_cell_id"], proseg_cell_metadata["cell"], strict=False))
    # copying the transcripts dataframe to avoid modifying the original
    xenium_transcripts = xenium_transcripts.copy()
    # adding the mapped ID into the dataframe
    xenium_transcripts["cell_id_numeric"] = xenium_transcripts["cell_id"].map(cell_id_dict).astype("Int64")

    st.fs.create_spatialdata(
        points=xenium_transcripts,
        images=xenium_image,
        shapes=proseg_cell_boundaries,
        coord_columns=["x_location", "y_location", "z_location"],
        cell_key_points="cell_id_numeric",
        cell_key_shapes="cell",
        relabel_points=True,
        relabel_shapes=True,
        consolidate_shapes=True,
    )
