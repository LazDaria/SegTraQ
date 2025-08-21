import segtraq as st


def test_create_geopandas_df(
    xenium_image,
    xenium_cell_labels,
    xenium_nucleus_labels,
    xenium_cell_boundaries,
    xenium_nucleus_boundaries,
    xenium_transcripts,
):
    # converting the data frames into geopandas dfs
    st.fs.create_geopandas_df(xenium_cell_boundaries)
    st.fs.create_geopandas_df(xenium_nucleus_boundaries)


def test_create_spatialdata(
    xenium_image,
    xenium_cell_labels,
    xenium_nucleus_labels,
    xenium_cell_boundaries,
    xenium_nucleus_boundaries,
    xenium_transcripts,
):
    # converting the data frames into geopandas dfs
    cell_shapes = st.fs.create_geopandas_df(xenium_cell_boundaries)
    nucleus_shapes = st.fs.create_geopandas_df(xenium_nucleus_boundaries)

    sdata = st.fs.create_spatialdata(
        points=xenium_transcripts,
        labels={"cell_labels": xenium_cell_labels, "nucleus_labels": xenium_nucleus_labels},
        images=xenium_image,
        shapes={"cell_boundaries": cell_shapes, "nucleus_boundaries": nucleus_shapes},
        # optional, if your coordinates are not named x, y, z
        coord_columns=["x_location", "y_location", "z_location"],
        consolidate_shapes=True,
    )

    st.fs.validate_spatialdata(sdata, cell_key_points="cell_id")
