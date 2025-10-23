import copy
import gzip
import shutil
import warnings
from collections.abc import Sequence
from pathlib import Path

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
import tifffile as tiff
import xarray as xr
from rasterio.features import rasterize
from scipy.sparse import csr_matrix
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.validation import make_valid
from skimage.measure import find_contours


def build_cell_polygons_from_vertices(
    df: pd.DataFrame,
    id_col: str = "label_id",
    x_col: str = "vertex_x",
    y_col: str = "vertex_y",
    keep_attrs: Sequence[str] = ("cell_id",),
    drop_closed_duplicate: bool = True,
    fix_invalid: bool = True,
) -> gpd.GeoDataFrame:
    """
    Convert a vertex table (one row per boundary vertex) into a GeoDataFrame of cell polygons.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with at least [id_col, x_col, y_col] columns.
    id_col : str, default="label_id"
        Column name identifying cells (e.g. segmentation label ID).
    x_col : str, default="vertex_x"
        Column containing vertex x-coordinates (microns).
    y_col : str, default="vertex_y"
        Column containing vertex y-coordinates (microns).
    keep_attrs : sequence of str, default=("cell_id",)
        Additional columns to retain in the output. For each cell,
        the first row value is taken.
    drop_closed_duplicate : bool, default=True
        If True, drop the trailing vertex if it duplicates the first
        (common in Xenium files).
    fix_invalid : bool, default=True
        If True, attempt to repair invalid polygons using `shapely.make_valid`.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame indexed by `id_col`, with columns `[id_col, keep_attrs..., geometry]`.
        Each row corresponds to one cell polygon. Invalid or empty geometries are skipped.
    """
    rows = []
    geometries = []

    for label_id, group in df.groupby(id_col, sort=False):
        xs = group[x_col].to_numpy()
        ys = group[y_col].to_numpy()

        # Drop last duplicate closing point if present
        if drop_closed_duplicate and len(xs) >= 2 and xs[0] == xs[-1] and ys[0] == ys[-1]:
            xs = xs[:-1]
            ys = ys[:-1]

        if len(xs) < 3:
            continue  # not enough points to form a polygon

        poly = Polygon(np.column_stack([xs, ys]))

        # Repair invalid polygons
        if fix_invalid and not poly.is_valid:
            poly = make_valid(poly)
            if poly.geom_type == "GeometryCollection":
                polys = [p for p in poly.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
                if not polys:
                    continue
                poly = polys[0] if len(polys) == 1 else MultiPolygon(polys)

        if poly.is_empty:
            continue

        row = {id_col: int(label_id) if pd.notna(label_id) else None}
        for attr in keep_attrs:
            if attr in group.columns:
                row[attr] = group[attr].iloc[0]

        rows.append(row)
        geometries.append(poly)

    gdf = gpd.GeoDataFrame(rows, geometry=geometries)
    gdf.set_index(id_col, inplace=True)
    return gdf


def labels_mode_projection(stack: np.ndarray) -> np.ndarray:
    """
    Compute a mode projection of a 3D label image along the Z axis.

    For each pixel (x, y), the label most frequently occurring across
    Z slices is assigned. Background (0) is ignored in the count.

    Parameters
    ----------
    stack : np.ndarray
        Integer label array of shape (Z, H, W).

    Returns
    -------
    np.ndarray
        2D array of shape (H, W) with the modal label per pixel.
    """
    if stack.ndim != 3:
        raise ValueError("Input stack must be 3D (Z, H, W).")

    Z, H, W = stack.shape
    flat = stack.reshape(Z, -1).T  # (H*W, Z)
    max_label = int(stack.max())
    counts = np.zeros((flat.shape[0], max_label + 1), dtype=np.int32)
    idx = np.arange(flat.shape[0])

    # Count occurrences per pixel
    for z in range(Z):
        np.add.at(counts, (idx, flat[:, z]), 1)

    counts[:, 0] = 0  # ignore background
    winner = counts.argmax(axis=1).astype(np.int32)
    return winner.reshape(H, W)


def labels_to_shapes(label_img: np.ndarray, simplify_tolerance: float | None = 0.5) -> gpd.GeoDataFrame:
    """
    Convert a 2D label image into polygon boundaries.

    Each connected label is represented by one Polygon or MultiPolygon.

    Parameters
    ----------
    label_img : np.ndarray
        2D array of integer labels. Background should be 0.
    simplify_tolerance : float or None, default=0.5
        Simplification tolerance for polygon boundaries. Set to None or 0
        to disable simplification.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with columns ["cell_id", "label_id", "geometry"],
        indexed by "label_id". Each row corresponds to one labeled region.
    """
    if label_img.ndim != 2:
        raise ValueError("Input label_img must be 2D.")

    geometries = []
    label_ids = []

    unique_labels = np.unique(label_img)
    unique_labels = unique_labels[unique_labels != 0]  # skip background

    for lid in unique_labels:
        mask = (label_img == lid).astype(np.uint8)
        contours = find_contours(mask, 0.5)

        polys = []
        for contour in contours:
            poly = Polygon(np.c_[contour[:, 1], contour[:, 0]])
            if not poly.is_valid or poly.area == 0:
                continue
            if simplify_tolerance and simplify_tolerance > 0:
                poly = poly.simplify(simplify_tolerance, preserve_topology=True)
            polys.append(poly)

        if not polys:
            continue

        geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
        geometries.append(geom)
        label_ids.append(int(lid))

    gdf = gpd.GeoDataFrame({"cell_id": label_ids, "label_id": label_ids, "geometry": geometries}).set_index("label_id")

    return gdf


def make_adata(X, obs: pd.DataFrame, var: pd.DataFrame, X_est: np.ndarray | None = None) -> ad.AnnData:
    """
    Build an AnnData object with optional estimated counts.

    Parameters
    ----------
    X : array-like or csr_matrix
        Expression matrix (cells x genes).
    obs : pd.DataFrame
        Cell metadata. Index should be cell IDs.
    var : pd.DataFrame
        Gene metadata. Index should be gene symbols.
    X_est : array-like or csr_matrix, optional
        Estimated expression values (same shape as X).

    Returns
    -------
    AnnData
        Annotated data matrix with optional "X_estimated" layer.
    """
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    # Ensure indices are strings to avoid ImplicitModificationWarning
    obs = obs.copy()
    obs.index = obs.index.astype(str)

    var = var.copy()
    var.index = var.index.astype(str)

    adata = ad.AnnData(X=X, obs=obs, var=var)

    if X_est is not None:
        adata.layers["X_estimated"] = csr_matrix(X_est)

    return adata


def read_dapi_image(path: Path) -> np.ndarray:
    """
    Read a DAPI image from TIFF and ensure correct dimensions.

    Parameters
    ----------
    path : Path
        Path to the DAPI TIFF file.

    Returns
    -------
    np.ndarray
        DAPI image with shape (c, y, x). Adds a channel axis if necessary.
    """
    dapi = tiff.imread(path)
    if dapi.ndim == 2:  # add channel axis if single-channel
        dapi = dapi[None, ...]
    return dapi


def read_labels(cell_mask_path: Path, nuc_mask_path: Path) -> dict[str, np.ndarray]:
    """
    Read cell and nucleus label masks from TIFF.

    Parameters
    ----------
    cell_mask_path : Path
        Path to the cell mask TIFF.
    nuc_mask_path : Path
        Path to the nucleus mask TIFF.

    Returns
    -------
    dict of {str: np.ndarray}
        Dictionary with keys 'cell_labels' and 'nucleus_labels'.
    """
    cell_labels = tiff.imread(cell_mask_path)
    nucleus_labels = tiff.imread(nuc_mask_path)
    return {"cell_labels": cell_labels, "nucleus_labels": nucleus_labels}


def read_shapes(path: Path, build_from_vertices: bool = True, backend: str = "pd"):
    """
    Read cell or nucleus shapes from Parquet or GeoJSON.

    Parameters
    ----------
    path : Path
        Path to a parquet or geojson file with shape definitions.
    build_from_vertices : bool, default=True
        If True and file is parquet, use build_cell_polygons_from_vertices.

    Returns
    -------
    GeoDataFrame
        Shape geometries indexed by label ID.
    """
    if path.suffix == ".parquet":
        if backend == "pd":
            df = pd.read_parquet(path)
        elif backend == "gpd":
            df = gpd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Please use 'pd' or 'gpd'.")
        if build_from_vertices:
            return build_cell_polygons_from_vertices(df)
        return df
    else:
        return gpd.read_file(path)


def read_transcripts(path: Path) -> pd.DataFrame:
    """
    Read transcript coordinates and attributes from Parquet or CSV.

    Parameters
    ----------
    path : Path
        Path to transcripts file (.parquet, .csv, or .csv.gz).

    Returns
    ----------
    pd.DataFrame
        DataFrame containing transcript data.
    """

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="gzip" if path.suffix.endswith(".gz") else None)

    return df


def make_points(
    df: pd.DataFrame, rename_map: dict[str, str] | None = None, uint32_max_placeholder: int | None = None
) -> pd.DataFrame:
    """
    Create a standardized transcript dataframe from a file.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing transcript data.
    rename_map : dict, optional
        Optional column renaming dictionary.
    uint32_max_placeholder : int, optional
        If provided, replace this value with 0 in integer columns.

    Returns
    -------
    pd.DataFrame
        Transcript dataframe with standardized columns:
        ['x', 'y', 'z', 'feature_name', 'cell_id', ...]
    """

    if rename_map is not None:
        df = df.rename(columns=rename_map)

    # Standardize coordinate columns
    for old, new in [("x_location", "x"), ("y_location", "y"), ("z_location", "z")]:
        if old in df.columns:
            df = df.rename(columns={old: new})

    if "feature_name" in df.columns:
        df["feature_name"] = df["feature_name"].astype("category")

    if "is_gene" in df.columns:
        df["is_gene"] = df["is_gene"].astype("str")

    if uint32_max_placeholder is not None:
        # uint32_max placeholder meaning “no assignment / background”
        df.loc[df["cell_id"] == uint32_max_placeholder, "cell_id"] = 0

    # replacing NaN cell_id with 0 (background)
    if "cell_id" in df.columns:
        # if cell_id is categorical or string, replace NaN with "UNASSIGNED"
        if df["cell_id"].dtype.name == "category" or df["cell_id"].dtype == object or df["cell_id"].dtype == str:
            df["cell_id"] = df["cell_id"].fillna("UNASSIGNED")
        else:
            df["cell_id"] = df["cell_id"].fillna(0)

    return df


def decompress_geojson(gz_path: Path) -> Path:
    """
    Decompress a .geojson.gz to .geojson if needed.

    Parameters
    ----------
    gz_path : Path
        Path to the compressed geojson.gz file.

    Returns
    -------
    Path
        Path to the decompressed .geojson file.
    """
    if gz_path.suffix == ".gz":
        json_path = gz_path.with_suffix("")  # strip .gz
        if not json_path.exists():
            with gzip.open(gz_path, "rt") as f_in, open(json_path, "w") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return json_path
    return gz_path


def build_spatialdata_from_proseg(
    adata: ad.AnnData,
    path_to_10xdata: Path,
    path_to_proseg_data: Path,
    polygons_gdf: gpd.GeoDataFrame,
    consolidate_shapes: bool = True,
) -> sd.SpatialData:
    """
    Builds a SpatialData object from processed segmentation and transcriptomics data.
    This function integrates segmentation polygons, DAPI images, nucleus masks, and transcript metadata
    to construct a comprehensive SpatialData object suitable for spatial transcriptomics analysis.
    It processes multi-layer polygon data, rasterizes cell boundaries, projects label stacks to 2D,
    and incorporates transcript and nucleus information.
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix containing gene expression and cell metadata.
    path_to_10xdata : Path
        Path to the directory containing 10x Genomics image and mask files.
    path_to_proseg_data : Path
        Path to the directory containing processed segmentation and transcriptomics data.
    polygons_gdf : gpd.GeoDataFrame
        GeoDataFrame containing cell segmentation polygons with layer and cell identifiers.
    consolidate_shapes : bool, optional
        Whether to consolidate shapes across layers into a single shape layer (default is True).
    Returns
    -------
    sd.SpatialData
        A SpatialData object containing integrated images, labels, shapes, transcript points, and tables.
    Notes
    -----
    - The function expects specific file names in the provided directories:
        - DAPI image: "dapi_um.tif"
        - Nucleus mask: "nuc_mask_um.tif"
        - Nucleus boundaries: "nucleus_boundaries.parquet"
        - Transcript metadata: "transcript-metadata.csv.gz"
    - Cell and nucleus labels are rasterized and projected to 2D for downstream analysis.
    """
    dapi = read_dapi_image(path_to_10xdata / "dapi_um.tif")
    H, W = dapi.shape[1:]

    # Polygon layers → labels + shapes
    labels_dict, shapes_dict = {}, {}
    z_levels = sorted(polygons_gdf["layer"].unique())
    stack = np.zeros((len(z_levels), H, W), dtype=np.uint32)

    for zi, z in enumerate(z_levels):
        # select the current layer first
        layer_gdf = polygons_gdf[polygons_gdf["layer"] == z]

        # then filter out empty or missing geometries
        layer_gdf = layer_gdf[~layer_gdf.geometry.is_empty & layer_gdf.geometry.notna()]

        # Shapes
        layer_shapes = layer_gdf.set_index("cell")["geometry"].to_frame().copy()
        layer_shapes.index.name = "label_id"
        layer_shapes.index += 1
        layer_shapes["cell_id"] = layer_shapes.index
        shapes_dict[f"cell_boundaries_z{int(z)}"] = layer_shapes

        # Labels via rasterize
        shapes_iter = (
            (mapping(geom), int(cid) + 1) for cid, geom in zip(layer_gdf["cell"], layer_gdf.geometry, strict=False)
        )
        img = rasterize(shapes_iter, out_shape=(H, W), fill=0, dtype=np.uint32)
        labels_dict[f"cell_labels_z{int(z)}"] = img
        stack[zi] = img

    # Projection to 2D
    proj = labels_mode_projection(stack)
    labels_dict["cell_labels"] = proj
    shapes_dict["cell_boundaries"] = labels_to_shapes(proj, simplify_tolerance=0.5)

    # Nucleus
    nucleus_labels = tiff.imread(path_to_10xdata / "nuc_mask_um.tif")
    labels_dict["nucleus_labels"] = nucleus_labels
    nucleus_shapes = read_shapes(path_to_10xdata / "nucleus_boundaries.parquet")
    shapes_dict["nucleus_boundaries"] = nucleus_shapes

    # Transcripts
    transcripts_df = read_transcripts(path_to_proseg_data / "transcript-metadata.csv.gz")
    transcripts_df = transcripts_df.rename(columns={"assignment": "cell_id", "gene": "feature_name"})
    transcripts_df["cell_id"] = transcripts_df["cell_id"].fillna(0)
    transcripts_df["cell_id"] = (transcripts_df["cell_id"] + 1).astype(int)
    transcripts = make_points(
        transcripts_df,
        uint32_max_placeholder=2**32,
    )

    # Assemble SpatialData
    sdata = create_spatialdata(
        points=transcripts,
        labels=labels_dict,
        shapes=shapes_dict,
        tables=adata,
        images=dapi,
        background_cell_id=0,
        consolidate_shapes=consolidate_shapes,
    )
    return sdata


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
    shapes : pd.DataFrame, dictionary, or None, optional
        DataFrame containing cell boundary polygons. Must include the column specified by `cell_key_shapes`.
        If a dictionary is provided, it should contain a key 'cell_boundaries' with the DataFrame.
        Default is None.
    labels : np.ndarray or None, optional
        Segmentation label image (2D or 3D array) with cell IDs. Default is None.
        If a dictionary is provided, it should contain a key 'cell_labels' with the DataFrame.
    tables : pd.DataFrame, AnnData or None, optional
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
            if background_cell_id != 0:
                assert points[cell_key_points].min() >= 1, (
                    "Cell IDs in points must start at 1. "
                    f"Found minimum cell ID: {points[cell_key_points].min()}. "
                    f"If you want to relabel the points by adding 1, set relabel_points=True. "
                    f"Alternatively, if your unassigned transcripts are labeled with 0, "
                    f"you can set background_cell_id=0."
                )
        else:
            points = points.copy()  # avoid modifying the original DataFrame
            points[cell_key_points] = points[cell_key_points] + 1

    # === SHAPES (POLYGONS) ===
    shapes_sd = None
    shapes_sd_dict = dict()

    assert shapes is None or isinstance(shapes, pd.DataFrame) or isinstance(shapes, dict), (
        "Shapes must be a pandas DataFrame or dictionary or None"
    )

    # if shapes is a dictionary, we assess if it contains valid keys
    if isinstance(shapes, dict):
        assert "cell_boundaries" in shapes.keys(), "Shapes dictionary must contain key: 'cell_boundaries'."
        # if there are multiple keys (e. g. from a nuclear and a whole cell mask), we only check the whole cell mask
        # the other one goes into the spatialdata dict directly
        other_keys = set(shapes.keys()) - {"cell_boundaries"}
        if len(other_keys) > 0:
            for key in other_keys:
                shapes_sd = sd.models.ShapesModel.parse(shapes[key])
                shapes_sd_dict[key] = shapes_sd

            shapes = shapes["cell_boundaries"]  # use the cell boundaries for further processing

    if shapes is not None:
        assert cell_key_shapes in shapes.columns, (
            f"Shapes DataFrame must contain column: {cell_key_shapes}. "
            f"Available columns: {shapes.columns.tolist()}. "
            f"If you want to use a different column, set the cell_key_shapes parameter."
        )
        shapes_cell_ids = set(shapes[cell_key_shapes])

        # if the cell IDs in shapes are integer-based, we check that they start at 1
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
        missing_in_polygons = transcript_ids - shapes_cell_ids - {background_cell_id}
        if not consolidate_shapes:
            assert not missing_in_polygons, (
                f"Missing {len(missing_in_polygons)} cell IDs from polygons: {missing_in_polygons}. "
                f"If you want to consolidate the shapes and the transcripts, set consolidate_shapes=True. "
                f"This will set the cell IDs of these transcripts to {background_cell_id}. "
                f"You can change this by setting the background_cell_id parameter."
            )
        elif len(missing_in_polygons) > 0:
            # relabel points with missing cell IDs to background_cell_id
            points[cell_key_points] = points[cell_key_points].apply(
                lambda x: background_cell_id if x in missing_in_polygons else x
            )
            warnings.warn(
                f"Missing {len(missing_in_polygons)} cell IDs from shapes: {missing_in_polygons}. "
                f"These cells are present in the points, but not in the shapes. "
                f"The points have been relabeled to {background_cell_id} (unassigned).",
                UserWarning,
                stacklevel=2,
            )

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
            if len(shapes_sd_dict) == 0:
                shapes_sd_dict = {"cell_boundaries": shapes_sd}
            else:
                shapes_sd_dict["cell_boundaries"] = shapes_sd

    # === LABELS ===
    labels_sd = None
    labels_sd_dict = dict()

    assert labels is None or isinstance(labels, np.ndarray) or isinstance(labels, dict), (
        "Labels must be a numpy array or dictionary or None"
    )

    # if labels is a dictionary, we assess if it contains valid keys
    if isinstance(labels, dict):
        assert "cell_labels" in labels.keys(), "Labels dictionary must contain key: 'cell_labels'."
        # if there are multiple keys (e. g. from a nuclear and a whole cell mask), we only check the whole cell mask
        # the other one goes into the spatialdata dict directly
        if len(labels.keys()) > 0:
            for key in labels.keys():
                labels_sd = sd.models.Labels2DModel.parse(labels[key], dims=["y", "x"])
                labels_sd_dict[key] = labels_sd

    # The code is checking if the variable `labels` is not `None`. If `labels` is not `None`, the code
    # block following the `if` statement will be executed.
    if labels is not None and len(labels_sd_dict) == 0:
        labels_sd = sd.models.Labels2DModel.parse(labels, dims=["y", "x"])
        labels_sd_dict = {"cell_labels": labels_sd}

    # === TABLES ===
    tables_sd = None
    if tables is not None:
        if isinstance(tables, ad.AnnData):
            adata = tables
            if adata.obs[cell_key_tables].dtype.kind in "iu":
                if not relabel_tables:
                    assert adata.obs[cell_key_tables].min() >= 1, (
                        f"Cell IDs in tables must start at 1. "
                        f"Found minimum cell ID: {adata.obs[cell_key_tables].min()}. "
                        f"If you want to relabel the tables by adding 1, set relabel_tables=True."
                    )
                else:
                    adata = copy.deepcopy(adata)  # avoid modifying the original AnnData
                    adata.obs[cell_key_tables] = adata.obs[cell_key_tables] + 1
        else:
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

        if "region" not in adata.obs.columns:
            adata.obs["region"] = pd.Categorical(["cell_labels"] * len(adata))
        if "label_id" not in adata.obs.columns:
            adata.obs["label_id"] = adata.obs_names.astype("int")

        # check that all cells in points are present in the tables
        missing_in_tables = set(points[cell_key_points]) - set(adata.obs["cell_id"]) - {background_cell_id}
        if not consolidate_tables:
            assert not missing_in_tables, (
                f"Missing {len(missing_in_tables)} cell IDs from tables: {missing_in_tables}. "
                f"If you want to consolidate the tables and the transcripts, set consolidate_tables=True. "
                f"This will remove the missing cell IDs from the points."
            )
        elif len(missing_in_tables) > 0:
            points = points[~points[cell_key_points].isin(missing_in_tables)]
            # checking if points is empty after removing missing cell IDs
            assert len(points) > 0, (
                "No points left after consolidating with tables. "
                "Please check your data to ensure that your cell IDs in tables match the ones in points."
            )

        tables_sd = sd.models.TableModel.parse(
            adata, region_key="region", region="cell_labels", instance_key="label_id"
        )

    # === IMAGES ===
    images_sd = None
    if images is not None:
        if images.ndim == 2:
            # If images are 2D, we need to expand dimensions to fit the Image2DModel
            images = np.expand_dims(images, axis=0)
        images_sd = sd.models.Image2DModel.parse(images, dims=["c", "y", "x"])

    # we only add these at the end of the method to ensure that the points are relabeled and filtered correctly
    points_sd = sd.models.PointsModel.parse(points)

    # Generate spatial data object
    sdata = sd.SpatialData(
        images={"image": images_sd} if images is not None else {},
        points={"transcripts": points_sd},
        shapes=shapes_sd_dict,
        tables={"table": tables_sd} if tables is not None else {},
        labels=labels_sd_dict,
    )

    # === FINAL VALIDATION ===
    validate_spatialdata(
        sdata,
        shape_key="cell_boundaries",  # list(shapes_sd_dict.keys()),
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
    cell_key_points: str = "cell_id",
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
            missing_in_shapes = tables_cell_ids - shapes_cell_ids - {background_cell_id}
            missing_in_tables = shapes_cell_ids - tables_cell_ids - {background_cell_id}
            if len(missing_in_tables) != 0:
                warnings.warn(
                    f"Missing {len(missing_in_tables)} cell IDs in tables: {missing_in_tables}. "
                    "These cells are present in shapes, but not in tables. "
                    "This might lead to inconsistencies in the spatialdata object.",
                    stacklevel=2,
                )
            if len(missing_in_shapes) != 0:
                warnings.warn(
                    f"Missing {len(missing_in_shapes)} cell IDs in shapes: {missing_in_shapes}. "
                    "These cells are present in tables, but not in shapes. "
                    "This might lead to inconsistencies in the spatialdata object.",
                    stacklevel=2,
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
                f"There are {len(shapes_cell_ids)} cell IDs in shapes, but only {len(labels_cell_ids)} are in labels. "
                f"This might lead to inconsistencies in the spatialdata object.",
                stacklevel=2,
            )
        if num_missing_in_shapes > 0:
            warnings.warn(
                f"Missing {num_missing_in_shapes} cell IDs in shapes: "
                f"There are {len(labels_cell_ids)} cell IDs in labels, but only {len(shapes_cell_ids)} are in shapes. "
                f"This might lead to inconsistencies in the spatialdata object.",
                stacklevel=2,
            )

    return True
