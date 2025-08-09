import warnings

import anndata as ad
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from shapely.geometry import Polygon
from skimage import measure


def create_spatialdata(
    points: pd.DataFrame,
    shapes=None,
    labels=None,
    tables=None,
    images=None,
    cell_key_points="assignment",
    cell_key_shapes="cell_id",
    cell_key_tables="cell_id",
    shape_layer_key="layer",
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

    # === SHAPES (POLYGONS) ===
    shapes_sd = None
    shapes_sd_dict = dict()

    assert shapes is None or isinstance(shapes, pd.DataFrame), "Shapes must be a pandas DataFrame or None"

    if shapes is not None:
        assert cell_key_shapes in shapes.columns, (
            f"Shapes DataFrame must contain column: {cell_key_shapes}. "
            f"Available columns: {shapes.columns.tolist()}. "
            f"If you want to use a different column, set the cell_key_shapes parameter."
        )
        shapes_cell_ids = set(shapes[cell_key_shapes])

        if not relabel_shapes:
            if shapes is not None:
                assert shapes[cell_key_shapes].min() >= 1, (
                    f"Cell IDs in shapes must start at 1. "
                    f"Found minimum cell ID: {shapes[cell_key_shapes].min()}. "
                    f"If you want to relabel the shapes by adding 1, set relabel_shapes=True."
                )
        else:
            shapes = shapes.copy()  # avoid modifying the original DataFrame
            shapes[cell_key_shapes] = shapes[cell_key_shapes] + 1

        transcript_ids = set(points[cell_key_points].unique())
        missing_in_polygons = transcript_ids - shapes_cell_ids
        if not consolidate_shapes:
            assert not missing_in_polygons, (
                f"Missing {len(missing_in_polygons)} cell IDs from polygons: {missing_in_polygons}. "
                "If you want to consolidate the shapes and the transcripts, set consolidate_shapes=True. "
                "This will remove the missing cell IDs from the points."
            )
        elif len(missing_in_polygons) > 0:
            # remove points that are not in the polygons
            points = points[~points[cell_key_points].isin(missing_in_polygons)]

        # check if shapes contains cell IDs that occur multiple times
        # if there are, this likely means that there are multiple layers that should be split into separate polygons
        if shapes[cell_key_shapes].duplicated().any():
            assert shape_layer_key in shapes.columns, (
                f"Some cell IDs in shapes occur multiple times. "
                f"This is likely due to multiple z layers being present in your shapes (e. g. when using ProSeg). "
                f"To split these into separate polygons, set the shape_layer_key parameter. "
                f"Available columns: {shapes.columns.tolist()}"
            )
            for i, layer in enumerate(shapes[shape_layer_key].unique()):
                layer_shapes = shapes[shapes[shape_layer_key] == layer]
                shapes_sd_dict[f"cell_boundaries_layer_{i}"] = sd.models.ShapesModel.parse(layer_shapes)
        else:
            shapes_sd = sd.models.ShapesModel.parse(shapes)
            shapes_sd_dict = {"cell_boundaries": shapes_sd}

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
        table_metadata = list(table_metadata)
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

    # we only add these at the end of the method to ensure that the points are relabeled and filtered correctly
    points_sd = sd.models.PointsModel.parse(points)

    # Generate spatial data object
    sdata = sd.SpatialData(
        images={"image": images_sd} if images is not None else {},
        points={"transcripts": points_sd},
        shapes=shapes_sd_dict,
        tables={"table": tables_sd} if tables is not None else {},
        labels={"cell_labels": labels_sd} if labels is not None else {},
    )

    # === FINAL VALIDATION ===
    validate_spatialdata(
        sdata,
        shape_key=list(shapes_sd_dict.keys()),
        label_key="cell_labels",
        points_key="transcripts",
        table_key="table",
        cell_key_points=cell_key_points,
        cell_key_shapes=cell_key_shapes,
        cell_key_tables=cell_key_tables,
        data_key=None,
    )

    return sdata


def validate_spatialdata(
    sdata: sd.SpatialData,
    shape_key: str | list[str] = "cell_boundaries",
    label_key: str = "cell_labels",
    points_key: str = "transcripts",
    table_key: str = "table",
    cell_key_points: str = "assignment",
    cell_key_shapes: str = "cell_id",
    cell_key_tables: str = "cell_id",
    data_key: str = None,
) -> bool:
    """
    Validate the given SpatialData object.

    Args:
        spatial_data (sd.SpatialData): The SpatialData object to validate.

    Returns:
        bool: True if the SpatialData object is valid, False otherwise.
    """
    if not isinstance(sdata, sd.SpatialData):
        raise TypeError("Input must be an instance of sd.SpatialData")

    contains_points = len(sdata.points) > 0
    contains_shapes = len(sdata.shapes) > 0
    contains_labels = len(sdata.labels) > 0
    contains_tables = len(sdata.tables) > 0

    # check if there are points in the spatial data
    if not contains_points:
        raise ValueError("SpatialData object must contain points (transcripts)")

    # get the cell IDs from the points
    points = sdata.points[points_key]
    assert cell_key_points in points.columns, (
        f"Points DataFrame must contain column to identify cells: {cell_key_points}. "
        f"Available columns: {points.columns.tolist()}. "
        f"If you want to use a different column, set the cell_key_points parameter."
    )
    transcript_ids = set(points[cell_key_points].unique())
    shapes_cell_ids = set()
    labels_cell_ids = set()

    # if there are shapes, ensure that there are no cell IDs in the points that are not in the shapes
    if contains_shapes:
        # we can have multiple shape keys (e. g. when using multiple layers in proseg), so we need to handle them here
        if isinstance(shape_key, str):
            assert shape_key in sdata.shapes, (
                f"Shapes DataFrame must contain key: {shape_key}. "
                f"Available keys: {list(sdata.shapes.keys())}. "
                f"If you want to use a different key, set the shape_key parameter."
            )
            shapes = sdata.shapes[shape_key]
        elif isinstance(shape_key, list):
            # if multiple shape keys are provided, we need to check each one
            shapes = pd.concat([sdata.shapes[key] for key in shape_key], ignore_index=True)
        else:
            raise ValueError("shape_key must be a string or a list of strings")

        assert cell_key_shapes in shapes.columns, (
            f"Shapes DataFrame must contain column: {cell_key_shapes}. "
            f"Available columns: {shapes.columns.tolist()}. "
            f"If you want to use a different column, set the cell_key_shapes parameter."
        )
        shapes_cell_ids = set(shapes[cell_key_shapes])
        missing_in_polygons = transcript_ids - shapes_cell_ids
        assert len(missing_in_polygons) == 0, (
            f"Missing {len(missing_in_polygons)} cell IDs from polygons: {missing_in_polygons}. "
            f"These cell IDs are present in the points, but not in the shapes."
        )

        # if shapes and tables are present, ensure that the cell IDs match
        # checking that the adata and the polygons have the same cell IDs
        if contains_tables:
            assert table_key in sdata.tables, (
                f"Tables DataFrame must contain key: {table_key}. "
                f"Available keys: {list(sdata.tables.keys())}. "
                f"If you want to use a different key, set the table_key parameter."
            )
            table = sdata.tables[table_key]
            assert cell_key_tables in table.obs.columns, (
                f"Tables DataFrame must contain column: {cell_key_tables}. "
                f"Available columns: {table.obs.columns.tolist()}. "
                f"If you want to use a different column, set the cell_key_tables parameter."
            )
            tables_cell_ids = set(table.obs[cell_key_tables].values)
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

    # if there are labels, ensure that there are no cell IDs in the points that are not in the labels
    if contains_labels:
        labels = sdata.labels[label_key]

        # handling weird spatialdata structures
        if isinstance(labels, xr.DataTree):
            assert data_key is not None, (
                f"It looks like your labels are stored as a DataTree. "
                f"Please provide a data_key to access the labels data. Available keys are: {list(labels.keys())}."
            )
            assert data_key.split("/")[0] in labels.keys(), (
                f"Data key {data_key} not found in the labels data. Available keys: {list(labels.keys())}"
            )

            labels = labels[data_key]  # Get the dataset node

            assert isinstance(labels, xr.DataArray), (
                f"The labels data should be a DataArray. Please provide a valid data key. "
                f"Available keys are: {[data_key + '/' + x for x in list(labels.keys())]}."
            )

        labels_cell_ids = set(np.unique(labels)) - {0}  # Exclude background label (0)
        missing_in_labels = transcript_ids - labels_cell_ids
        assert len(missing_in_labels) == 0, (
            f"Missing {len(missing_in_labels)} cell IDs from labels: "
            f"{list(missing_in_labels)[: min(5, len(missing_in_labels))]}... "
            f"These cell IDs are present in the points, but not in the labels. "
            f"The cell IDs in your labels look like this: {list(labels_cell_ids)[: min(5, len(labels_cell_ids))]}... "
        )

    # if there are both shapes and labels, ensure they are compatible
    if contains_shapes and contains_labels:
        missing_in_shapes = labels_cell_ids - shapes_cell_ids
        missing_in_labels = shapes_cell_ids - labels_cell_ids
        if len(missing_in_labels) > 0:
            warnings.warn(
                f"Missing {len(missing_in_labels)} cell IDs in labels: "
                f"{list(missing_in_labels)[: min(5, len(missing_in_labels))]}... "
                f"These cells are present in shapes, but not in labels. "
                f"Cell IDs in labels look like this: {list(labels_cell_ids)[: min(5, len(labels_cell_ids))]}... "
                f"This might lead to inconsistencies in the spatialdata object.",
                stacklevel=2,
            )
        if len(missing_in_shapes) > 0:
            warnings.warn(
                f"Missing {len(missing_in_shapes)} cell IDs in shapes: "
                f"{list(missing_in_shapes)[: min(5, len(missing_in_shapes))]}... "
                f"These cells are present in labels, but not in shapes. "
                f"Cell IDs in shapes look like this: {list(shapes_cell_ids)[: min(5, len(shapes_cell_ids))]}... "
                f"This might lead to inconsistencies in the spatialdata object.",
                stacklevel=2,
            )


def compute_shapes(
    sdata: sd.SpatialData, labels_key: str = "labels", shape_key: str = "cell_boundaries"
) -> sd.SpatialData:
    assert shape_key not in sdata.shapes, (
        f"Shapes with key '{shape_key}' already exist in SpatialData. "
        "Please choose a different key by setting the shape_key parameter or remove the existing shapes."
    )

    # Ensure labels are present
    assert labels_key in sdata.labels, (
        f"Labels DataFrame must contain key: {labels_key}. "
        f"Available keys: {list(sdata.labels.keys())}. "
        f"If you want to use a different key, set the labels_key parameter."
    )

    # Get numpy array from sdata.labels
    labels = (
        sdata.labels[labels_key].values if hasattr(sdata.labels[labels_key], "values") else sdata.labels[labels_key]
    )

    # Ensure it's an integer mask
    labels = np.asarray(labels, dtype=np.int32)

    polygons = []
    cell_ids = []

    for cell_id in np.unique(labels):
        if cell_id == 0:  # skip background
            continue

        # Find contours for this cell (connectivity 1)
        contours = measure.find_contours(labels == cell_id, level=0.5)
        for contour in contours:
            # contour coordinates are (row, col), flip to (x, y)
            poly = Polygon(contour[:, ::-1])
            if not poly.is_valid or poly.is_empty:
                continue
            polygons.append(poly)
            cell_ids.append(cell_id)

    # Build DataFrame with cell_id and geometry
    shapes_df = pd.DataFrame({"cell_id": cell_ids, "geometry": polygons})

    # Convert DataFrame to a GeoDataFrame if SpatialData expects it
    try:
        import geopandas as gpd

        shapes_gdf = gpd.GeoDataFrame(shapes_df, geometry="geometry", crs="EPSG:4326")
    except ImportError:
        shapes_gdf = shapes_df

    # Add to SpatialData
    sdata.shapes[shape_key] = sd.models.ShapesModel.parse(shapes_gdf)

    return sdata
