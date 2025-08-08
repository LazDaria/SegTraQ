import warnings

import anndata as ad
import numpy as np
import pandas as pd
import spatialdata as sd


def create_spatialdata(
    points: pd.DataFrame,
    shapes=None,
    labels=None,
    tables=None,
    images=None,
    shape_key="cell_boundaries",
    label_key="cell_labels",
    gene_key="gene_name",
    cell_key_points="assignment",
    cell_key_shapes="cell_id",
    cell_key_tables="cell_id",
    relabel_points: bool = False,
    relabel_shapes: bool = False,
    relabel_tables: bool = False,
    table_metadata=("cell_id", "centroid_x", "centroid_y", "cell_size"),
    consolidate_shapes: bool = False,
    consolidate_tables: bool = False,
    consolidate_labels: bool = True,
) -> sd.SpatialData:
    assert isinstance(points, pd.DataFrame), "Points must be a pandas DataFrame"
    required_columns = ["x", "y", "z"]
    assert all(col in points.columns for col in required_columns), (
        f"Points DataFrame must contain columns: {required_columns}"
    )

    # === POINTS (TRANSCRIPTS) ===
    assert cell_key_points in points.columns, (
        f"Points DataFrame must contain column: {cell_key_points}. "
        f"Available columns: {points.columns.tolist()}. "
        f"If you want to use a different column, set the cell_key_points parameter."
    )
    # check that the minimum cell ID is 1
    if not relabel_points:
        assert points[cell_key_points].min() >= 1, (
            "Cell IDs in points must start at 1. "
            f"Found minimum cell ID: {points[cell_key_points].min()}. "
            f"If you want to relabel the points by adding 1, set relabel_points=True."
        )
    else:
        points = points.copy()  # avoid modifying the original DataFrame
        points[cell_key_points] = points[cell_key_points] + 1
    points_sd = sd.models.PointsModel.parse(points)

    # === SHAPES (POLYGONS) ===
    shapes_sd = None
    # == 2D ==
    assert shapes is None or isinstance(shapes, pd.DataFrame), "Shapes must be a pandas DataFrame or None"
    if shapes is not None:
        assert cell_key_shapes in shapes.columns, (
            f"Shapes DataFrame must contain column: {cell_key_shapes}. "
            f"Available columns: {shapes.columns.tolist()}. "
            f"If you want to use a different column, set the cell_key_shapes parameter."
        )
        poly_ids = set(shapes[cell_key_shapes])

    if not relabel_shapes:
        if shapes is not None:
            assert shapes[cell_key_shapes].min() >= 1, (
                f"Cell IDs in shapes must start at 1. "
                f"Found minimum cell ID: {shapes[cell_key_shapes].min()}. "
                f"If you want to relabel the shapes by adding 1, set relabel_shapes=True."
            )
    else:
        if shapes is not None:
            shapes = shapes.copy()  # avoid modifying the original DataFrame
            shapes[cell_key_shapes] = shapes[cell_key_shapes] + 1

    transcript_ids = set(points[cell_key_points].unique())
    missing_in_polygons = transcript_ids - poly_ids
    if not consolidate_shapes:
        assert not missing_in_polygons, (
            f"Missing {len(missing_in_polygons)} cell IDs from polygons: {missing_in_polygons}. "
            "If you want to consolidate the shapes and the transcripts, set consolidate_shapes=True. "
            "This will remove the missing cell IDs from the points."
        )
    elif len(missing_in_polygons) > 0:
        # remove points that are not in the polygons
        points = points[~points[cell_key_points].isin(missing_in_polygons)]

    if shapes is not None:
        shapes_sd = sd.models.ShapesModel.parse(shapes)

    # == 3D ==
    # TODO: implement 3D shapes (probably sensible to split up the layers into different layers within the shapes)

    # === LABELS ===
    labels_sd = None
    if labels is not None:
        # consistency checks
        # with the points
        cell_ids_from_points = set(points[cell_key_points].unique())
        cell_ids_from_labels = set(np.unique(labels))
        missing_in_labels = cell_ids_from_points - cell_ids_from_labels
        if not consolidate_labels:
            assert not missing_in_labels, (
                f"Missing {len(missing_in_labels)} cell IDs from labels: {missing_in_labels}. "
                f"If you want to consolidate the labels and the transcripts, set consolidate_labels=True. "
                f"This will remove the missing cell IDs from the points."
            )
        elif len(missing_in_labels) > 0:
            points = points[~points[cell_key_points].isin(missing_in_labels)]

        labels_sd = sd.models.Labels2DModel.parse(labels)

    # === TABLES ===
    tables_sd = None
    if tables is not None:
        # Prepare obs DataFrame with string index
        obs_df = tables[table_metadata].copy()
        obs_df.index = obs_df.index.astype(str)  # ensure AnnData-compatible index

        # Prepare X matrix and var names
        X_df = tables.drop(columns=table_metadata)
        var_names = [str(col) for col in X_df.columns]  # force string var names

        # Create AnnData
        adata = ad.AnnData(
            X=X_df.values,
            obs=obs_df,
        )
        adata.var_names = var_names

        if not relabel_tables:
            assert tables[cell_key_tables].min() >= 1, (
                f"Cell IDs in tables must start at 1. "
                f"Found minimum cell ID: {tables[cell_key_tables].min()}. "
                f"If you want to relabel the tables by adding 1, set relabel_tables=True."
            )
        else:
            tables = tables.copy()  # avoid modifying the original DataFrame
            tables[cell_key_tables] = tables[cell_key_tables] + 1

        adata.obs["region"] = pd.Categorical(["cell_labels"] * len(adata))
        adata.obs["mask_id"] = adata.obs_names.astype("int")

        # check that all cells in points are present in the tables
        missing_in_tables = set(points[cell_key_points]) - set(adata.obs["mask_id"])
        if not consolidate_tables:
            assert not missing_in_tables, (
                f"Missing {len(missing_in_tables)} cell IDs from tables: {missing_in_tables}. "
                f"If you want to consolidate the tables and the transcripts, set consolidate_tables=True. "
                f"This will remove the missing cell IDs from the points."
            )
        elif len(missing_in_tables) > 0:
            points = points[~points[cell_key_points].isin(missing_in_tables)]

        tables_sd = sd.models.TableModel.parse(adata, region_key="region", region="cell_labels", instance_key="mask_id")

    # === IMAGES ===
    images_sd = None
    if images is not None:
        if images.ndim == 2:
            # If images are 2D, we need to expand dimensions to fit the Image2DModel
            images = np.expand_dims(images, axis=0)
        images_sd = sd.models.Image2DModel.parse(images)

    # === FINAL VALIDATION ===
    # TODO: should put this into a separate function, so that it can run on already existing SpatialData objects as well
    # checking that the adata and the polygons have the same cell IDs
    if shapes_sd is not None and tables_sd is not None:
        shapes_cell_ids = set(shapes_sd[cell_key_shapes].values)
        tables_cell_ids = set(tables_sd.obs[cell_key_tables].values)
        missing_in_shapes = tables_cell_ids - shapes_cell_ids
        missing_in_tables = shapes_cell_ids - tables_cell_ids
        assert len(missing_in_tables) == 0, (
            f"Missing {len(missing_in_tables)} cell IDs in tables: {missing_in_tables}. "
            "These cells are present in shapes, but not in tables. "
            "This might lead to inconsistencies in the spatialdata object."
        )
        assert len(missing_in_shapes) == 0, (
            f"Missing {len(missing_in_shapes)} cell IDs in shapes: {missing_in_shapes}. "
            "These cells are present in tables, but not in shapes. "
            "This might lead to inconsistencies in the spatialdata object."
        )

    # checking that the polygons and the labels have the same cell IDs
    if shapes_sd is not None and labels is not None:
        shapes_cell_ids = set(shapes_sd[cell_key_shapes].values)
        labels_cell_ids = set(np.unique(labels))
        missing_in_shapes = labels_cell_ids - shapes_cell_ids
        missing_in_labels = shapes_cell_ids - labels_cell_ids
        if len(missing_in_labels) > 0:
            warnings.warn(
                f"Missing {len(missing_in_labels)} cell IDs in labels: {missing_in_labels}. "
                f"These cells are present in shapes, but not in labels. "
                f"This might lead to inconsistencies in the spatialdata object.",
                stacklevel=2,
            )
        if len(missing_in_shapes) > 0:
            warnings.warn(
                f"Missing {len(missing_in_shapes)} cell IDs in shapes: {missing_in_shapes}. "
                f"These cells are present in labels, but not in shapes. "
                f"This might lead to inconsistencies in the spatialdata object.",
                stacklevel=2,
            )

    # Generate spatial data object
    sdata = sd.SpatialData(
        images={"image": images_sd} if images is not None else {},
        points={"points": points_sd},
        shapes={"cell_boundaries": shapes_sd} if shapes is not None else {},
        tables={"table": tables_sd} if tables is not None else {},
        labels={"cell_labels": labels_sd} if labels is not None else {},
    )

    return sdata


def validate_spatialdata(
    sdata: sd.SpatialData,
    shape_key: str = "cell_boundaries",
    label_key: str = "cell_labels",
    transcript_key: str = "points",
    cell_key: str = "cell_id",
) -> bool:
    """
    Validate the given SpatialData object.

    Args:
        spatial_data (sd.SpatialData): The SpatialData object to validate.

    Returns:
        bool: True if the SpatialData object is valid, False otherwise.
    """
    raise NotImplementedError("THIS FUNCTION IS NOT IMPLEMENTED YET: validate_spatialdata")
    # if not isinstance(sdata, sd.SpatialData):
    #     raise TypeError("Input must be an instance of sd.SpatialData")

    # contains_points = len(sdata.points) > 0
    # contains_shapes = len(sdata.shapes) > 0
    # contains_labels = len(sdata.labels) > 0

    # # check if there are points in the spatial data
    # if not contains_points:
    #     raise ValueError("SpatialData object must contain points _points)")

    # # check if there is either a shapes or a labels attribute
    # if not contains_shapes and not contains_labels:
    #     # TODO: implement a check for shapes or labels
    #     raise ValueError(
    #         "SpatialData object must contain a segmentation, either as shapes or as labels"
    #     )

    # # if there are both shapes and labels, ensure they are compatible
    # if contains_shapes and contains_labels:
    #     raise NotImplementedError(
    #         "THIS CHECK IS NOT IMPLEMENTED YET: SpatialData object cannot
    # contain both shapes and labels at the same time"
    #     )

    # # if there are shapes, ensure that there are no cell IDs in the points that are not in the shapes
    # if contains_shapes:
    #     cell_ids_from_points = np.unique(
    #         sdata.points[transcript_key][cell_key].values.compute()
    #     )
    #     cell_ids_from_shapes = np.unique(
    #         sdata.shapes[shape_key][cell_key].values.compute()
    #     )
    #     # if not np.all(np.isin(cell_ids_from_points, cell_ids_from_shapes)):
    #     # raise ValueError(
    #     #    f"Some cell IDs in points are not present in shapes.
    #     #    Found {len(cell_ids_from_points)} cell IDs in points, but only {len(cell_ids_from_shapes)} in shapes.
    #     #    {len(cell_ids_from_points) - len(cell_ids_from_shapes)} could not be assigned to a shape."
    #     # )

    # # if there are labels, ensure that there are no cell IDs in the points that are not in the labels
    # if contains_labels:
    #     label_ids = {label.id for label in sdata.labels}
    #     for point in sdata.points:
    #         if point.cell_id is not None and point.cell_id not in label_ids:
    #             raise ValueError(
    #                 f"Point with cell ID {point.cell_id} not found in labels"
    #             )

    # return True
