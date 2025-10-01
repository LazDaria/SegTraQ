from collections.abc import Sequence
from pathlib import Path

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.sparse import csr_matrix
from shapely.geometry import MultiPolygon, Polygon
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


def read_shapes(path: Path, build_from_vertices: bool = True):
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
        df = pd.read_parquet(path)
        if build_from_vertices:
            return build_cell_polygons_from_vertices(df)
        return df
    else:
        return gpd.read_file(path)


def read_transcripts(path: Path, rename_map: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Read transcript coordinates and attributes from Parquet or CSV.

    Parameters
    ----------
    path : Path
        Path to transcripts file (.parquet, .csv, or .csv.gz).
    rename_map : dict, optional
        Optional column renaming dictionary.

    Returns
    -------
    pd.DataFrame
        Transcript dataframe with standardized columns:
        ['x', 'y', 'z', 'feature_name', 'cell_id', ...]
    """
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="gzip" if path.suffix.endswith(".gz") else None)

    if rename_map is not None:
        df = df.rename(columns=rename_map)

    # Standardize coordinate columns
    for old, new in [("x_location", "x"), ("y_location", "y"), ("z_location", "z")]:
        if old in df.columns:
            df = df.rename(columns={old: new})

    if "feature_name" in df.columns:
        df["feature_name"] = df["feature_name"].astype("category")

    return df
