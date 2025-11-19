import warnings

import numpy as np
import pandas as pd
import spatialdata as sd
import geopandas as gpd
from geopandas import GeoDataFrame
from pandas import Series
from rtree.index import Index
from shapely.geometry.base import BaseGeometry
from spatialdata.models import PointsModel
from scipy.spatial import cKDTree
from ..utils import _looks_like_counts


def _compute_iou(poly1: BaseGeometry, poly2: BaseGeometry) -> float:
    """Compute IoU between two shape polygons."""

    if not (poly1.is_valid and poly2.is_valid):  # TODO - make polygons valid later
        return np.nan
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    return inter_area / union_area if union_area > 0 else 0.0


def _process_cell(
    cell_row: Series,
    shapes_cell_id_key: str | None,
    id_key: str | None,
    nuc_boundaries: GeoDataFrame,
    nuc_sindex: Index,
) -> dict[str | int, str | int, int | None | float]:
    """For one cell polygon compute the IoU with the best-matching nucleus."""

    cell_geom = cell_row.geometry

    cell_id = cell_row[shapes_cell_id_key] if shapes_cell_id_key is not None else cell_row.name

    # Get candidate nuclei bounding boxes that overlap this cell's bbox
    candidate_idx = list(nuc_sindex.intersection(cell_geom.bounds))

    if not candidate_idx:
        return {id_key: cell_row.name, "best_nuc_id": np.nan, "IoU": 0.0}

    candidates = nuc_boundaries.iloc[candidate_idx]

    best_iou: float = 0.0
    best_nuc_id = np.nan
    for _, nuc in candidates.iterrows():
        nuc_geom = nuc.geometry
        iou = _compute_iou(cell_geom, nuc_geom)
        if pd.notna(iou) and iou > best_iou:
            best_iou = iou
            best_nuc_id = nuc.name

    return {id_key: cell_id, "best_nuc_id": best_nuc_id, "IoU": best_iou}


def _shapes_by_feature_df(
    sdata: sd.SpatialData,
    points_key: str = "transcripts",
    shapes_key: str = "nucleus_boundaries",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_z_key: str | None = "z",
) -> pd.DataFrame:
    """
    Aggregate feature counts per region (nucleus or other), converting transcripts to 2D if needed.

    Parameters
    ----------
        sdata : SpatialData
            A `SpatialData` object containing segmented and transcript-assigned spatial
            transcriptomics data (images, tables, points, shapes and optional labels).
        shapes_key : str, default="nucleus_boundaries"
            Key in `sdata.shapes` for defining the regions to aggregate by.
        points_x_key : str, default="x"
            Column for the x-coordinate of each transcript/spot.
        points_gene_key : str, default="feature_name"
            Column specifying the gene/feature name for each transcript/spot.
        points_x_key : str, default="x"
            Column for the x-coordinate of each transcript/spot.
        points_y_key : str, default="y"
            Column for the y-coordinate of each transcript/spot.
        points_z_key : str or None, optional, default="z"
            Column for the z-coordinate (3D data). If `None`, data are treated as 2D.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by shapes ID, columns = features (genes/proteins), values = counts.
    """

    pts = sdata.points[points_key]
    # check dimensionality: assume 3D if "z" in actual data columns
    df = pts.compute()
    is_3d = points_z_key in df.columns  # TODO - maybe there is a better way to check if transcripts are 3D

    if is_3d:
        transcripts_2d_key = points_key + "_2D"
        df2 = df.drop(columns=[points_z_key])
        coord_sys = "global"  # TODO find an soft coded way to get coordinate system of transcripts
        trans = sd.transformations.get_transformation(pts, to_coordinate_system=coord_sys, get_all=False)

        if hasattr(trans, "scale") and hasattr(trans, "axes"):
            # reduce transformation to 2D to avoid shape mismatch error
            trans.scale = trans.scale[:2]
            trans.axes = trans.axes[:2]

        trans_dict = {coord_sys: trans}

        if not df2.index.is_unique:
            warnings.warn(
                "Index of sdata.points[points_key] is not unique — resetting index to avoid reindexing errors.",
                UserWarning,
                stacklevel=2,
            )
            df2 = df2.reset_index(drop=True)

        pts2 = PointsModel.parse(
            df2,
            name=transcripts_2d_key,
            coordinates={"x": points_x_key, "y": points_y_key},
            transformations=trans_dict,
        )
        sdata.points[transcripts_2d_key] = pts2
        value_key = transcripts_2d_key
    else:
        value_key = points_key

    # perform aggregation
    sdata2 = sdata.aggregate(
        values=value_key,
        by=shapes_key,
        value_key=points_gene_key,
        agg_func="count",
        deep_copy=False,
    )
    ad = sdata2.tables["table"]
    X = ad.X
    arr = X.toarray() if hasattr(X, "toarray") else X
    df_out = pd.DataFrame(arr, index=sdata2[shapes_key].index, columns=ad.var_names)
    return df_out

def _get_center_and_border_shapes(
    sdata: sd.SpatialData,
    shapes_key: str = "cell_boundaries",
    shapes_cell_id_key: str = "cell_id",
    tables_cell_id_key: str = "cell_id",
    erosion_fraction_of_radius: float | None = 0.4,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Create eroded 'center' and 'border' shapes for each cell by shrinking the
    original cell polygons and taking the difference.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with cell boundary polygons in `sdata.shapes[shapes_key]`.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons.
    shapes_cell_id_key : str, default="cell_id"
        Column name linking shapes to cell IDs.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    erosion_fraction_of_radius : float or None, default=0.4
        Fraction of the equivalent radius to use as erosion
        Example: 0.4 means erode by 40% of the radius.

    Returns
    -------
    center_gdf : GeoDataFrame
        GeoDataFrame with eroded "center" polygons, indexed or labeled by cell_id.
    border_gdf : GeoDataFrame
        GeoDataFrame with "border" polygons (cell minus center), same indexing.
    """
    cells_gdf = sdata.shapes[shapes_key].copy()

    if shapes_cell_id_key is not None:
        id_key = shapes_cell_id_key
        cells_gdf.set_index(id_key, drop=True)
    elif cells_gdf.index.name is not None:
        id_key = cells_gdf.index.name
    else:
        id_key = tables_cell_id_key
        cells_gdf.index.name = id_key

    center_records = []
    border_records = []

    areas = cells_gdf.geometry.area
    # Avoid weird issues with tiny/empty shapes
    areas = areas.clip(lower=1e-6)
    radii = np.sqrt(areas / np.pi)
    erosion_dists = radii * erosion_fraction_of_radius

    for _, row in cells_gdf.iterrows():
        cid = row.name
        geom = row.geometry

        # Erode polygon to get center
        d = erosion_dists.loc[cid]
        center_geom = geom.buffer(-d)
        if center_geom.is_empty:
            # if erosion kills the polygon, treat center as empty and border as full cell
            center_geom = None
            border_geom = geom
        else:
            # border = full cell minus center
            border_geom = geom.difference(center_geom)

        center_records.append({id_key: cid, "geometry": center_geom})
        border_records.append({id_key: cid, "geometry": border_geom})

    center_gdf = gpd.GeoDataFrame(center_records, geometry="geometry", crs=cells_gdf.crs)
    border_gdf = gpd.GeoDataFrame(border_records, geometry="geometry", crs=cells_gdf.crs)

    # Optional: store in sdata.shapes if you like
    # sdata.shapes["cell_centers"] = center_gdf[~center_gdf.isna()]
    # sdata.shapes["cell_borders"] = border_gdf[~border_gdf.isna()]

    return center_gdf[~center_gdf.isna()], border_gdf[~border_gdf.isna()]


def _compute_ncvs_within_radius(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    shapes_cell_id_key: str = "cell_id",
    radius_factor: float = 2.0,
) -> pd.DataFrame:
    """
    Compute neighborhood composition vectors (NCVs) as the average gene expression
    of neighboring cells within a user-defined distance.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with cell shapes and cell-level table.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level AnnData.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons.
    shapes_cell_id_key : str, default="cell_id"
        Column in the shapes GeoDataFrame linking polygons to cell IDs.
    radius_factor : float, default=2.0
        Neighborhood radius factor in the same coordinate units as the shapes.

    Returns
    -------
    pandas.DataFrame
        DataFrame of NCVs: index = cell IDs, columns = genes, values = average
        expression of neighbors within `radius` (excluding the focal cell).
    """
    # Get centroids for each cell shape
    cells_gdf = sdata.shapes[shapes_key].copy()
    if shapes_cell_id_key is not None:
        id_key = shapes_cell_id_key
        cells_gdf = cells_gdf.set_index(id_key, drop=True)
    elif cells_gdf.index.name is not None:
        id_key = cells_gdf.index.name
    else:
        id_key = tables_cell_id_key
        cells_gdf.index.name = id_key

    tbl = sdata.tables[tables_key]
    ad = tbl

    # Get cell expression matrix (use raw if available and X is not counts)
    X = ad.X
    # Check if X looks like counts
    if _looks_like_counts(X):
        arr = X.toarray() if hasattr(X, "toarray") else X
    elif "raw" not in tbl.layers:
        raise ValueError(
            f"'raw' layer does not exist in sdata.tables['{tables_key}'], "
            "and the main matrix does not look like counts."
        )
    else:
        raw = tbl.layers["raw"]
        arr = raw.toarray() if hasattr(raw, "toarray") else raw

    mask = ad.obs[tables_cell_id_key].isin(cells_gdf.index)
    expr_cells = pd.DataFrame(
        arr[mask, :],
        index=ad.obs[tables_cell_id_key][mask],
        columns=ad.var.index,
    )

    # Align order between shapes and table: we assume one shape per cell
    cells_gdf = cells_gdf.loc[expr_cells.index]
    centroids = cells_gdf.geometry.centroid
    coords = np.vstack([centroids.x.values, centroids.y.values]).T

    areas = cells_gdf.geometry.area
    # Avoid weird issues with tiny/empty shapes
    areas = areas.clip(lower=1e-6)
    radii = np.sqrt(areas / np.pi)

    tree = cKDTree(coords)

    n_cells = expr_cells.shape[0]
    genes = expr_cells.columns
    ncv_arr = np.zeros_like(expr_cells.values, dtype=float)

    for i in range(n_cells):
        # Query neighbors within radius (including itself)
        idxs = tree.query_ball_point(coords[i], r=radii[i]*radius_factor)
        # Remove self
        idxs = [j for j in idxs if j != i]
        if len(idxs) == 0:
            # no neighbors in radius: define NCV as zeros or NaN
            ncv_arr[i, :] = 0.0
        else:
            ncv_arr[i, :] = expr_cells.values[idxs, :].mean(axis=0)

    expr_ncv = pd.DataFrame(ncv_arr, index=expr_cells.index, columns=genes)
    return expr_ncv

