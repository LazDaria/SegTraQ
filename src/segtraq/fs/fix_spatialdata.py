import copy
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import spatialdata as sd
import geopandas as gpd
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon
from skimage import measure
from skimage.draw import polygon


def create_spatialdata(
    points: pd.DataFrame,
    shapes=None,
    labels=None,
    tables=None,
    images=None,
    cell_key_points="cell_id",
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
    background_cell_id: str = "UNASSIGNED",
    coord_columns: tuple[str, str, str] = ("x", "y", "z"),
) -> sd.SpatialData:
    """
    Creates a SpatialData object from provided spatial transcriptomics data components.

    This function integrates transcript (points), cell boundary (shapes), segmentation label (labels),
    cell feature (tables), and image data into a single SpatialData object, performing consistency checks
    and optional relabeling or consolidation of cell IDs across modalities.

    Parameters
    ----------
    points : pd.DataFrame
        DataFrame containing transcript coordinates and cell assignments.
        Must include the column specified by `cell_key_points`.
    shapes : pd.DataFrame or None, optional
        DataFrame containing cell boundary polygons. Must include the column specified by `cell_key_shapes`.
        Default is None.
    labels : np.ndarray or None, optional
        Segmentation label image (2D or 3D array) with cell IDs. Default is None.
    tables : pd.DataFrame or None, optional
        DataFrame containing per-cell features. Must include the column specified by `cell_key_tables`. Default is None.
    images : np.ndarray or None, optional
        Image data (2D or 3D array). Default is None.
    cell_key_points : str, optional
        Column name in `points` DataFrame indicating cell assignments. Default is "assignment".
    cell_key_shapes : str, optional
        Column name in `shapes` DataFrame indicating cell IDs. Default is "cell_id".
    cell_key_tables : str, optional
        Column name in `tables` DataFrame indicating cell IDs. Default is "cell_id".
    shape_layer_key : str, optional
        Column name in `shapes` DataFrame indicating layer information for splitting polygons. Default is "layer".
    relabel_points : bool, optional
        If True, increment all cell IDs in `points` by 1. Default is False.
    relabel_shapes : bool, optional
        If True, increment all cell IDs in `shapes` by 1. Default is False.
    relabel_tables : bool, optional
        If True, increment all cell IDs in `tables` by 1. Default is False.
    table_metadata : tuple of str, optional
        Column names in `tables` to use as metadata (obs) in AnnData.
        Default is ("cell_id", "centroid_x", "centroid_y", "cell_size").
    consolidate_shapes : bool, optional
        If True, remove points with cell IDs not present in `shapes`. Default is False.
    consolidate_tables : bool, optional
        If True, remove points with cell IDs not present in `tables`. Default is False.
    consolidate_labels : bool, optional
        If True, remove points with cell IDs not present in `labels`. Default is True.
    background_cell_id : str, optional
        Cell ID to use for unassigned transcripts in points. Default is "UNASSIGNED".
    coord_columns : tuple of str, optional
        Names of the coordinate columns in `points` DataFrame. Default is ("x", "y", "z").

    Returns
    -------
    sd.SpatialData
        A SpatialData object containing the integrated spatial transcriptomics data.

    Raises
    ------
    AssertionError
        If required columns are missing, cell IDs are inconsistent, or data integrity checks fail.

    Notes
    -----
    - Cell IDs in all modalities are expected to start at 1 unless relabeling is enabled.
    - If consolidation is enabled for shapes, tables, or labels, points with missing cell IDs in
    those modalities are removed.
    - For shapes with multiple layers, polygons are split by the `shape_layer_key` column.
    """
    assert isinstance(points, pd.DataFrame), "Points must be a pandas DataFrame"
    # check that x, y, and z coordinates are present in the points DataFrame
    assert all(col in points.columns for col in coord_columns), (
        f"Points DataFrame must contain columns: {coord_columns}. "
        f"Available columns: {points.columns.tolist()}. "
        f"If you want to use different columns for the coordinates, set the coord_columns parameter."
    )
    
    # if the coords_columns are not x, y, z, we relabel them
    if coord_columns != ("x", "y", "z"):
        points = points.rename(columns={coord_columns[0]: "x", coord_columns[1]: "y", coord_columns[2]: "z"})

    # === POINTS (TRANSCRIPTS) ===
    assert cell_key_points in points.columns, (
        f"Points DataFrame must contain column: {cell_key_points}. "
        f"Available columns: {points.columns.tolist()}. "
        f"If you want to use a different column, set the cell_key_points parameter."
    )
    # check that the minimum cell ID is 1 (if the cell IDs are integer-based)
    if points[cell_key_points].dtype.kind in "iu":
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

        if shapes[cell_key_shapes].dtype.kind in "iu":
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
        background_cell_id=background_cell_id,
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
    background_cell_id: str = "UNASSIGNED",
) -> bool:
    """
    Validates the integrity of a SpatialData object by checking the consistency of cell IDs across points,
    shapes, labels, and tables.

    This function ensures that:
    - All points have corresponding shapes, labels, and tables.
    - Cell IDs in points match those in shapes, labels, and tables.
    - If shapes or labels are present, they contain all cell IDs from the points.
    - If tables are present, they contain all cell IDs from the shapes.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object to validate.
    shape_key : str or list of str, optional
        Key(s) for accessing shapes in the SpatialData. Default is "cell_boundaries".
    label_key : str, optional
        Key for accessing labels in the SpatialData. Default is "cell_labels".
    points_key : str, optional
        Key for accessing points in the SpatialData. Default is "transcripts".
    table_key : str, optional
        Key for accessing tables in the SpatialData. Default is "table".
    cell_key_points : str, optional
        Column name in points DataFrame indicating cell assignments. Default is "assignment".
    cell_key_shapes : str, optional
        Column name in shapes DataFrame indicating cell IDs. Default is "cell_id".
    cell_key_tables : str, optional
        Column name in tables DataFrame indicating cell IDs. Default is "cell_id".
    data_key : str, optional
        Key for accessing data in labels if they are stored as a DataTree. Default is None.
    background_cell_id : str, optional
        Cell ID to use for unassigned transcripts in points. Default is "UNASSIGNED".

    Raises
    ------
    TypeError
        If the input is not an instance of sd.SpatialData.
    ValueError
        If the SpatialData object does not contain points or if there are inconsistencies in cell IDs.

    Returns
    -------
    bool
        True if the SpatialData object is valid, otherwise raises an error.
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
        missing_in_polygons = transcript_ids - shapes_cell_ids - {background_cell_id}
        assert len(missing_in_polygons) == 0, (
            f"Missing {len(missing_in_polygons)} cell IDs from polygons: {missing_in_polygons}. "
            f"These cell IDs are present in the points, but not in the shapes. "
            f"If your missing cell ID is indicating an unassigned transcript, "
            f"you can set the background_cell_id parameter."
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

        # label ID and cell ID are not the same
        labels_cell_ids = set(np.unique(labels)) - {0}  # Exclude background label (0)

    # if there are both shapes and labels, ensure they are compatible
    if contains_shapes and contains_labels:
        num_missing_in_shapes = len(labels_cell_ids) - len(shapes_cell_ids)
        num_missing_in_labels = len(shapes_cell_ids) - len(labels_cell_ids)
        if num_missing_in_labels > 0:
            warnings.warn(
                f"Missing {num_missing_in_labels} cell IDs in labels."
                f"There are {len(labels_cell_ids)} cell IDs in labels, but only {len(shapes_cell_ids)} are in shapes. "
                f"This might lead to inconsistencies in the spatialdata object.",
                stacklevel=2,
            )
        if num_missing_in_shapes > 0:
            warnings.warn(
                f"Missing {num_missing_in_shapes} cell IDs in shapes: "
                f"There are {len(shapes_cell_ids)} cell IDs in shapes, but only {len(labels_cell_ids)} are in labels. "
                f"This might lead to inconsistencies in the spatialdata object.",
                stacklevel=2,
            )

    return True


def compute_shapes(
    sdata: sd.SpatialData, labels_key: str = "labels", shape_key: str = "cell_boundaries"
) -> sd.SpatialData:
    """
    Compute cell shapes from the labels in the SpatialData object.
    This function extracts cell boundaries from the segmentation labels and stores them as polygons in
    the shapes of the SpatialData object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing segmentation labels.
    labels_key : str, optional
        Key for accessing the labels in the SpatialData. Default is "labels".
    shape_key : str, optional
        Key for storing the computed shapes in the SpatialData. Default is "cell_boundaries".

    Returns
    -------
    sd.SpatialData
        The updated SpatialData object with computed shapes added.

    Raises
    ------
    AssertionError
        If the labels are not present or if the shape_key already exists in the shapes of the SpatialData.
    TypeError
        If the labels are not in the expected format (e.g., not a numpy array or DataFrame).
    """
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


def compute_labels(
    sdata: sd.SpatialData,
    labels_key: str = "cell_labels",
    shapes_key: str = "cell_boundaries",
    cell_key_shapes: str = "cell_id",
) -> sd.SpatialData:
    """
    Compute labels from the shapes in the SpatialData object.
    This function generates a label array from the cell boundaries stored in the shapes of the SpatialData object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing cell boundaries.
    labels_key : str, optional
        Key for storing the generated labels in the SpatialData. Default is "cell_labels".
    shapes_key : str, optional
        Key for accessing the shapes in the SpatialData. Default is "cell_boundaries".
    cell_key_shapes : str, optional
        Key for accessing the cell IDs in the shapes. Default is "cell_id".

    Returns
    -------
    sd.SpatialData
        The updated SpatialData object with generated labels added.

    Raises
    ------
    AssertionError
        If the labels are not present or if the shapes_key does not exist in the shapes of the SpatialData.
    TypeError
        If the shapes are not in the expected format (e.g., not a DataFrame).
    """
    assert labels_key not in sdata.labels, (
        f"Labels with key '{labels_key}' already exist in SpatialData. "
        "Please choose a different key by setting the labels_key parameter or remove the existing labels."
    )

    # Ensure shapes are present
    assert shapes_key in sdata.shapes, (
        f"Shapes DataFrame must contain key: {shapes_key}. "
        f"Available keys: {list(sdata.shapes.keys())}. "
        f"If you want to use a different key, set the shapes_key parameter."
    )

    # Get shapes DataFrame
    shapes = sdata.shapes[shapes_key]
    assert cell_key_shapes in shapes.columns, (
        f"Shapes DataFrame must contain column: {cell_key_shapes}. "
        f"Available columns: {shapes.columns.tolist()}. "
        f"If you want to use a different column, set the cell_key_shapes parameter."
    )

    # if an image is present, we can use the image to figure out the size of the labels
    if len(sdata.images) > 0:
        image = next(iter(sdata.images.values())).squeeze()
        height, width = image.data.shape
    else:
        # if no image is present, we get the minimum and maximum coordinates from the shapes
        max_x = shapes["geometry"].apply(lambda geom: geom.bounds[2]).max()
        max_y = shapes["geometry"].apply(lambda geom: geom.bounds[3]).max()

        height = int(max_y)
        width = int(max_x)

    # Create an empty label array
    labels = np.zeros((height, width), dtype=np.int32)

    # Fill the label array with cell IDs from shapes
    for _, row in shapes.iterrows():
        cell_id = row[cell_key_shapes]
        geom = row["geometry"]

        # Ensure we have only polygons
        if isinstance(geom, Polygon):
            polygons = [geom]
        elif isinstance(geom, MultiPolygon):
            polygons = list(geom.geoms)
        else:
            continue  # skip unsupported geometries

        for poly in polygons:
            coords = np.array(poly.exterior.coords)
            rr, cc = polygon(coords[:, 1], coords[:, 0], shape=labels.shape)
            labels[rr, cc] = cell_id

    # copying the sdata object to avoid modifying the original
    # TODO: should make it possible to run this in-place
    sdata = copy.deepcopy(sdata)
    # Add labels to SpatialData
    sdata.labels[labels_key] = sd.models.Labels2DModel.parse(labels)

    return sdata


def compute_tables(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    shapes_key: str = "cell_boundaries",
    points_key: str = "transcripts",
    cell_key_shapes: str = "cell_id",
    cell_key_points: str = "assignment",
    gene_key: str = "gene",
) -> sd.SpatialData:
    """
    Compute tables from the shapes and points in the SpatialData object.
    This function generates a table of gene expression values for each cell based on the transcript points.
    It creates an AnnData object with the expression matrix and cell metadata, and stores it in the SpatialData object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing shapes and points.
    tables_key : str, optional
        Key for storing the generated tables in the SpatialData. Default is "table".
    shapes_key : str, optional
        Key for accessing the shapes in the SpatialData. Default is "cell_boundaries".
    points_key : str, optional
        Key for accessing the points in the SpatialData. Default is "transcripts".
    cell_key_shapes : str, optional
        Key for accessing the cell IDs in the shapes. Default is "cell_id".
    cell_key_points : str, optional
        Key for accessing the cell assignments in the points. Default is "assignment".
    gene_key : str, optional
        Key for accessing the gene names in the points. Default is "gene".

    Returns
    -------
    sd.SpatialData
        The updated SpatialData object with generated tables added.

    Raises
    ------
    AssertionError
        If the tables already exist or if required keys or columns are missing.
    TypeError
        If the shapes or points are not in the expected format (e.g., not a DataFrame).
    """
    assert tables_key not in sdata.tables, (
        f"Tables with key '{tables_key}' already exist in SpatialData. "
        f"Available tables: {list(sdata.tables.keys())}. "
        "Please choose a different key by setting the tables_key parameter or remove the existing table."
    )

    # Ensure shapes are present
    assert shapes_key in sdata.shapes, (
        f"Shapes DataFrame must contain key: {shapes_key}. "
        f"Available keys: {list(sdata.shapes.keys())}. "
        "If you want to use a different key, set the shapes_key parameter."
    )

    # Ensure points are present
    assert points_key in sdata.points, (
        f"Points DataFrame must contain key: {points_key}. "
        f"Available keys: {list(sdata.points.keys())}. "
        "If you want to use a different key, set the points_key parameter."
    )

    shapes = sdata.shapes[shapes_key]
    shapes = shapes.set_crs(None, allow_override=True)  # explicitly say “no CRS”
    points = sdata.points[points_key]

    # Check required columns in shapes
    assert cell_key_shapes in shapes.columns, (
        f"Shapes DataFrame must contain column: {cell_key_shapes}. "
        f"Available columns: {shapes.columns.tolist()}. "
        "If you want to use a different column, set the cell_key_shapes parameter."
    )

    # Check required columns in points
    assert cell_key_points in points.columns, (
        f"Points DataFrame must contain column: {cell_key_points}. "
        f"Available columns: {points.columns.tolist()}. "
        "If you want to use a different column, set the cell_key_points parameter."
    )
    assert gene_key in points.columns, (
        f"Points DataFrame must contain column: {gene_key}. "
        f"Available columns: {points.columns.tolist()}. "
        "If you want to use a different column, set the gene_key parameter."
    )

    # 1. Build expression matrix from points
    expr_df = (
        points[[cell_key_points, gene_key]]  # just the columns we need
        .compute()  # force into Pandas
        .groupby([cell_key_points, gene_key])
        .size()
        .unstack(fill_value=0)
    )

    # 2. Create obs from shapes
    obs_df = shapes.set_index(cell_key_shapes)[["geometry"]].copy()
    obs_df["cell_id"] = obs_df.index
    obs_df["centroid_x"] = obs_df.geometry.centroid.x
    obs_df["centroid_y"] = obs_df.geometry.centroid.y
    obs_df["cell_size"] = obs_df.geometry.area
    obs_df.drop(columns=["geometry"], inplace=True)

    # 3. Align obs and X
    all_cells = obs_df.index.union(expr_df.index)
    obs_df = obs_df.reindex(all_cells)
    expr_df = expr_df.reindex(all_cells, fill_value=0)
    expr_df.columns = expr_df.columns.astype(str)

    obs_df.index = obs_df.index.astype(str)
    expr_df.index = expr_df.index.astype(str)
    expr_df.columns = expr_df.columns.astype(str)

    # 4. Create AnnData
    adata = ad.AnnData(X=expr_df.to_numpy(), obs=obs_df, var=pd.DataFrame(index=expr_df.columns))

    # 5. Store in SpatialData
    sdata = copy.deepcopy(sdata)
    sdata.tables[tables_key] = adata
    return sdata


def create_geopandas_df(df: pd.DataFrame) -> gpd.GeoDataFrame:
    polygons = []
    ids = []

    assert 'cell_id' in df.columns, (
        "DataFrame must contain 'cell_id' column to group by cell IDs."
    )
    assert 'vertex_x' in df.columns and 'vertex_y' in df.columns, (
        "DataFrame must contain 'vertex_x' and 'vertex_y' columns for polygon coordinates."
    )

    for cell_id, group in df.groupby('cell_id'):
        # Group by label_id if you may have multiple polygons per cell
        polys = []
        for _, sub in group.groupby('label_id'):
            coords = list(zip(sub['vertex_x'], sub['vertex_y']))
            if len(coords) >= 3:  # valid polygon
                polys.append(Polygon(coords))
        if len(polys) == 1:
            polygons.append(polys[0])
        else:
            polygons.append(MultiPolygon(polys))
        ids.append(cell_id)

    return gpd.GeoDataFrame(
        {'cell_id': ids, 'geometry': polygons},
        crs="EPSG:4326"  # or your actual CRS
    )