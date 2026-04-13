import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from geopandas import GeoDataFrame
from pandas import Series
from rtree.index import Index
from scipy.spatial import cKDTree
from shapely.geometry.base import BaseGeometry

from ..utils import _is_background, _looks_like_counts, filter_cells


def _safe_intersection_area(poly1: BaseGeometry, poly2: BaseGeometry) -> float:
    if not (poly1.is_valid and poly2.is_valid):
        return np.nan
    return poly1.intersection(poly2).area


def _compute_iou_from_areas(inter_area: float, area1: float, area2: float) -> float:
    if np.isnan(inter_area) or area1 <= 0 or area2 <= 0:
        return np.nan
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else np.nan


def _compute_nucleus_fraction(inter_area: float, nuc_area: float) -> float:
    if np.isnan(inter_area) or nuc_area <= 0:
        return np.nan
    return inter_area / nuc_area


def _norm_log_df(df: pd.DataFrame, scale: float = 1e4) -> pd.DataFrame:
    # row-wise library size normalization + log1p
    sums = df.sum(axis=1).replace(0, np.nan)
    df_norm = df.div(sums, axis=0) * scale
    return np.log1p(df_norm).fillna(0.0)


def _process_cell(
    cell_row: Series,
    nucleus_shapes: GeoDataFrame,
    id_name: str,
    nuc_sindex: Index,
    select_by: str = "nucleus_fraction",  # "iou" or "nucleus_fraction"
    min_intersection_area: float = 0.0,  # optional filter to ignore tiny overlaps
) -> dict:
    """
    For one cell polygon, find the best-matching nucleus using either IoU or
    nucleus intersection fraction as the primary score.

    Tie-breaker: larger nucleus area.
    """
    if select_by not in ("iou", "nucleus_fraction"):
        raise ValueError(f"select_by must be 'iou' or 'nucleus_fraction', got {select_by!r}")

    cell_geom = cell_row.geometry
    cell_id = cell_row.name

    candidate_idx = list(nuc_sindex.intersection(cell_geom.bounds))

    # if there are no nuclei intersecting with our cell
    if not candidate_idx:
        return {
            id_name: cell_id,
            "nucleus_id": np.nan,
            "iou": np.nan,
            "nucleus_fraction": np.nan,
        }

    candidates = nucleus_shapes.iloc[candidate_idx]

    cell_area = cell_geom.area if cell_geom.is_valid else np.nan

    best = {
        "score": -np.inf,
        "nucleus_area": -np.inf,
        "intersection_area": -np.inf,
        "nucleus_id": np.nan,
        "iou": np.nan,
        "nucleus_fraction": np.nan,
    }

    for nucleus_id, nucleus in candidates.iterrows():
        nucleus_geom = nucleus.geometry
        if not (cell_geom.is_valid and nucleus_geom.is_valid):
            continue

        nucleus_area = nucleus_geom.area
        if nucleus_area <= 0 or cell_area <= 0:
            continue

        intersection_area = _safe_intersection_area(cell_geom, nucleus_geom)
        if np.isnan(intersection_area) or intersection_area <= min_intersection_area:
            continue

        iou = _compute_iou_from_areas(intersection_area, cell_area, nucleus_area)
        nucleus_fraction = _compute_nucleus_fraction(intersection_area, nucleus_area)

        score = iou if select_by == "iou" else nucleus_fraction

        # Compare with tie-breaks: score, then nucleus_area, then intersection_area, then nucleus_id
        better = (
            (score > best["score"])
            or (np.isclose(score, best["score"]) and nucleus_area > best["nucleus_area"])
            or (
                np.isclose(score, best["score"])
                and np.isclose(nucleus_area, best["nucleus_area"])
                and intersection_area > best["intersection_area"]
            )
            or (
                np.isclose(score, best["score"])
                and np.isclose(nucleus_area, best["nucleus_area"])
                and np.isclose(intersection_area, best["intersection_area"])
                and nucleus_id < best["nucleus_id"]
            )
        )

        if better:
            best.update(
                score=score,
                nucleus_area=nucleus_area,
                intersection_area=intersection_area,
                nucleus_id=nucleus_id,
                iou=iou,
                nucleus_fraction=nucleus_fraction,
            )

    # If nothing survived filtering
    if best["score"] == -np.inf:
        return {
            id_name: cell_id,
            "nucleus_id": np.nan,
            "iou": np.nan,
            "nucleus_fraction": np.nan,
        }

    return {
        id_name: cell_id,
        "nucleus_id": best["nucleus_id"],
        "iou": best["iou"],
        "nucleus_fraction": best["nucleus_fraction"],
    }


def _get_center_and_border_shapes(
    sdata: sd.SpatialData,
    shapes_key: str = "cell_boundaries",
    erosion_fraction_of_radius: float = 0.3,
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
    erosion_fraction_of_radius : float, default=0.3
        Fraction of the equivalent radius to use as erosion
        Example: 0.3 means erode by 30% of the radius.

    Returns
    -------
    center_gdf : GeoDataFrame
        GeoDataFrame with eroded "center" polygons, indexed or labeled by cell_id.
    border_gdf : GeoDataFrame
        GeoDataFrame with "border" polygons (cell minus center), same indexing.
    """
    cells_gdf = sdata.shapes[shapes_key].copy()

    id_key = cells_gdf.index.name

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

        # Erosion can't be done on invalid shapes
        if not geom.is_valid:
            center_geom = None
            border_geom = None
            continue

        # Erode polygon to get center
        d = erosion_dists.loc[cid]
        center_geom = geom.buffer(-d)

        if not center_geom.is_valid:  # erosion can lead to invalid shapes
            center_geom = None
            border_geom = None
        elif center_geom.is_empty:
            center_geom = None
            border_geom = geom
        else:
            border_geom = geom.difference(center_geom)

        # geom.difference can lead to invalid shapes
        if not border_geom.is_valid:
            border_geom = None
            center_geom = None

        center_records.append({id_key: cid, "geometry": center_geom})
        border_records.append({id_key: cid, "geometry": border_geom})

    center_gdf = gpd.GeoDataFrame(center_records, geometry="geometry", crs=cells_gdf.crs)
    border_gdf = gpd.GeoDataFrame(border_records, geometry="geometry", crs=cells_gdf.crs)

    center_gdf.set_index(id_key, drop=True, inplace=True)
    border_gdf.set_index(id_key, drop=True, inplace=True)

    return center_gdf[center_gdf.geometry.notna()], border_gdf[border_gdf.geometry.notna()]


def _get_filtered_points_df(
    sdata: sd.SpatialData,
    genes: str | list[str] | None,
    cell_type_key: str | None,
    cell_type_query: str | list[str] | None,
    tables_key: str,
    tables_cell_id_key: str,
    points_key: str,
    points_cell_id_key: str,
    points_gene_key: str,
    points_background_id: str,
) -> pd.DataFrame:
    tbl = sdata.tables[tables_key]
    pts = sdata.points[points_key]

    # subset to genes present in the table
    all_genes = pd.Index(tbl.var_names)
    pts = pts.dropna(subset=[points_gene_key])
    pts = pts[pts[points_gene_key].isin(all_genes)]

    # optionally subset to cell type of interest
    if cell_type_query is not None:
        query_vals = [cell_type_query] if isinstance(cell_type_query, str) else list(cell_type_query)
        adata = filter_cells(adata=tbl, col=cell_type_key, func=lambda x: x.isin(query_vals))
    else:
        adata = tbl
    cell_ids = adata.obs[tables_cell_id_key]

    # subset points to cells in tbl

    pts = pts[pts[points_cell_id_key].isin(cell_ids)]

    # optionally subset to gene selection
    if genes is not None:
        if isinstance(genes, str):
            pts = pts[pts[points_gene_key] == genes]
        else:
            pts = pts[pts[points_gene_key].isin(list(genes))]

    # remove background
    is_bg = _is_background(pts[points_cell_id_key], points_background_id)
    pts = pts.loc[~is_bg]

    # compute
    df = pts.compute() if hasattr(pts, "compute") else pts
    if df.empty:
        raise ValueError("No transcripts found after filtering.")

    return df


def _join_points_regions(
    sdata: sd.SpatialData,
    region_key: str,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_gene_key: str = "feature_name",
    points_cell_id_key: str = "cell_id",
    points_background_id: str = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    genes: str | list[str] | None = None,
    cell_type_key: str = "transferred_cell_type",
    cell_type_query: str | list[str] | None = None,
    predicate: str = "intersects",
    require_points_region_ID_match: bool = True,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Spatially join transcript points to region polygons and return:
      1) per-point region assignments, and
      2) a region x gene count matrix.

    This can be applied for nuclei, cell centers, cell borders, etc.

    The function:
      - filters background points and genes not present in `sdata.tables[tables_key].var_names`
      - converts points to a GeoDataFrame
      - performs a spatial join against `sdata.shapes[region_key]`
      - deduplicates points that intersect multiple polygons by keeping the first match
      - optionally keeps only points whose assigned region id equals points_cell_id_key
        (useful when region ids are cell ids, e.g. centers/borders; ensures compatibility
        with 3D-aware segmentation, where transcripts may share x/y coordinates but
        belong to different z-resolved cells)

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned spatial
        transcriptomics data (images, tables, points, shapes and optional labels).
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    region_key : str
        Key in `sdata.shapes` specifying which regions to use (e.g., `"nucleus_boundaries"`,
        `"cell_centers"`, `"cell_borders"`). Must contain a `geometry` column with polygons.
    points_key : str, default="transcripts"
        Key in `sdata.points` for spot/transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell (background).
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript/spot.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    genes : str | list[str] | None, optional
        String or list of strings indicating the feature/gene(s) to calculate the mean transcript coordiantes on.
        If None, all genes are used.
    cell_type_key : str
        Column in `sdata.tables[tables_key].obs` with cell-type labels.
    cell_type_query : str | list[str] | None, optional
        If provided, compute the metric only for cells whose `cell_type_key` matches these label(s).
    predicate: str, default="intersects"
        Spatial predicate passed to `geopandas.sjoin`.
        Common options: "intersects" (default), "within", "contains".
        For points-in-polygons, "within" is often appropriate; "intersects" includes boundary hits.
    require_points_region_ID_match: bool, default=True
        If not None, keep only points where the joined region id equals the values in this
        points column. Typical use: require_region_id_equals="cell_id" when `region_key`
        contains per-cell regions indexed by cell id (centers/borders).
        Set to None for nuclei (region ids are nucleus ids, not cell ids).

    Returns
    -------
    pts_joined : geopandas.GeoDataFrame
        Points with assigned region ids in column "region_id" (and geometry).
        Points that do not intersect any region have NA in "region_id" (because join is left).
    counts : pandas.DataFrame
        Region x gene count matrix (rows = all regions from shapes index, columns = all genes).
    """

    transcripts = _get_filtered_points_df(
        sdata=sdata,
        genes=genes,
        cell_type_key=cell_type_key,
        cell_type_query=cell_type_query,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_gene_key=points_gene_key,
        points_background_id=points_background_id,
    )

    cols = [points_cell_id_key, points_gene_key, points_x_key, points_y_key]
    transcripts = transcripts[cols]

    transcripts[points_gene_key] = transcripts[points_gene_key].cat.remove_unused_categories()

    # ensure we have a clean, unique point index for deduplication after sjoin
    # Dask indices are often non-unique (each partition starts at 0) - after.compute() duplicate indices persist
    if isinstance(transcripts, pd.DataFrame):
        if transcripts.index.is_unique:
            transcripts = transcripts.reset_index(drop=False).rename(columns={"index": "point_id"})
        else:
            transcripts = transcripts.reset_index(drop=True)
            transcripts["point_id"] = np.arange(len(transcripts), dtype=np.int64)

    pts_gdf = gpd.GeoDataFrame(
        transcripts,
        geometry=gpd.points_from_xy(transcripts[points_x_key], transcripts[points_y_key]),
        crs=sdata.shapes[region_key].crs,  # assume same CRS
    )[["point_id", points_cell_id_key, points_gene_key, "geometry"]]

    # prepare shapes/regions
    region_gdf = sdata.shapes[region_key].copy()
    all_regions = region_gdf.index

    # normalize region id into a plain column for join output clarity
    region_gdf.index.name = "region_id"
    region_gdf.reset_index(inplace=True)
    region_gdf = region_gdf[["region_id", "geometry"]]

    pts_joined = gpd.sjoin(
        pts_gdf,
        region_gdf,
        how="left",
        predicate=predicate,
    ).drop(columns=["index_right"])

    # if a point intersects multiple polygons, keep the first match
    pts_joined = pts_joined.sort_values("point_id").groupby("point_id", observed=True, as_index=False).first()

    # optionally restrict to points whose region id matches another point column
    if require_points_region_ID_match:
        pts_joined = pts_joined[pts_joined["region_id"] == pts_joined[points_cell_id_key]]

    # aggregate into region x gene counts
    all_genes = pd.Index(sdata.tables[tables_key].var_names)

    counts = (
        pts_joined[["region_id", points_gene_key]]
        .groupby(["region_id", points_gene_key], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=all_regions, columns=all_genes, fill_value=0)
    )

    return pts_joined, counts


def _compute_ncvs_within_radius(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    neighborhood_radius_factor: float = 2.0,
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
    neighborhood_radius_factor : float, default=2.0
        This is multiplied by each cell's radius to define the neighborhood distance for that cell.

    Returns
    -------
    pandas.DataFrame
        DataFrame of NCVs: index = cell IDs, columns = genes, values = average
        expression of neighbors within `radius` (excluding the focal cell).
    """
    # Get centroids for each cell shape
    cells_gdf = sdata.shapes[shapes_key].copy()

    ad = sdata.tables[tables_key]
    X = ad.X

    if _looks_like_counts(X):
        arr = X.toarray() if hasattr(X, "toarray") else X
    elif "counts" not in ad.layers:
        raise ValueError(
            f"'counts' layer does not exist in sdata.tables['{tables_key}'], "
            "and the main matrix does not look like counts."
        )
    else:
        counts = ad.layers["counts"]
        arr = counts.toarray() if hasattr(counts, "toarray") else counts

    mask = ad.obs[tables_cell_id_key].isin(cells_gdf.index)
    expr_cells = pd.DataFrame(
        arr[mask, :],
        index=ad.obs[tables_cell_id_key][mask],
        columns=ad.var_names,
    )

    # Align order between shapes and table: we assume one shape per cell
    cells_gdf = cells_gdf.loc[expr_cells.index]
    centroids = cells_gdf.geometry.centroid
    coords = np.vstack([centroids.x.values, centroids.y.values]).T

    areas = cells_gdf.geometry.area
    # Avoid weird issues with tiny/empty shapes
    areas = areas.clip(lower=1e-6)
    radii = np.sqrt(areas / np.pi)
    radii.reset_index(inplace=True, drop=True)

    tree = cKDTree(coords)

    n_cells = expr_cells.shape[0]
    genes = expr_cells.columns
    ncv_arr = np.zeros_like(expr_cells.values, dtype=float)

    for i in range(n_cells):
        # Query neighbors within radius (including itself)
        idxs = tree.query_ball_point(coords[i], r=radii[i] * neighborhood_radius_factor)
        # Remove self
        idxs = [j for j in idxs if j != i]
        if len(idxs) == 0:
            # no neighbors in radius: define NCV as zeros or NaN
            ncv_arr[i, :] = 0.0
        else:
            ncv_arr[i, :] = expr_cells.values[idxs, :].mean(axis=0)

    expr_ncv = pd.DataFrame(ncv_arr, index=expr_cells.index, columns=genes)
    return expr_ncv


def _get_center_border_counts(
    sdata,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    points_key: str = "transcripts",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_cell_id_key: str = "cell_id",
    points_background_id: str = "UNASSIGNED",
    erosion_fraction_of_radius: float = 0.3,
):
    center_gdf, border_gdf = _get_center_and_border_shapes(
        sdata=sdata,
        shapes_key=shapes_key,
        erosion_fraction_of_radius=erosion_fraction_of_radius,
    )

    cell_shape_transformation = sdata.shapes[shapes_key].attrs["transform"]

    sdata.shapes["cell_centers"] = sd.models.ShapesModel.parse(center_gdf, transformations=cell_shape_transformation)
    sdata.shapes["cell_borders"] = sd.models.ShapesModel.parse(border_gdf, transformations=cell_shape_transformation)

    tx_assigned_to_center, expr_center = _join_points_regions(
        sdata=sdata,
        region_key="cell_centers",
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        points_key=points_key,
        points_gene_key=points_gene_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        predicate="within",
    )

    tx_assigned_to_border, expr_border = _join_points_regions(
        sdata=sdata,
        region_key="cell_borders",
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        points_key=points_key,
        points_gene_key=points_gene_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        predicate="within",
    )

    # checking if there are any transcripts that were counted in both center and border
    # this should never be the case, hence we issue a warning if it happens
    center_transcripts = tx_assigned_to_center["point_id"].values
    border_transcripts = tx_assigned_to_border["point_id"].values
    intersecting_transcripts = set(center_transcripts).intersection(set(border_transcripts))
    if len(intersecting_transcripts) > 0:
        warnings.warn(
            f"{len(intersecting_transcripts)} transcripts were counted in both center and border regions. "
            f"Please report this issue to the SegTraQ developers.",
            UserWarning,
            stacklevel=2,
        )

    return expr_center, expr_border

def _build_count_vectors(
    gene_codes: np.ndarray,
    region_codes: np.ndarray,
    n_genes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert transcript-level rows into per-region gene count vectors.

    Parameters
    ----------
    gene_codes : np.ndarray
        Integer gene indices for each transcript row.
    region_codes : np.ndarray
        Region assignment for each transcript row:
            0 = center
            1 = border
            2 = neighborhood
    n_genes : int
        Total number of genes in the fixed gene universe.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Count vectors for center, border, and neighborhood.
    """
    x_center = np.bincount(gene_codes[region_codes == 0], minlength=n_genes)
    x_border = np.bincount(gene_codes[region_codes == 1], minlength=n_genes)
    x_neighborhood = np.bincount(gene_codes[region_codes == 2], minlength=n_genes)
    return x_center, x_border, x_neighborhood

# def _find_neighbors_by_distance(
#     sdata,
#     tables_key: str = "table",
#     tables_cell_id_key: str = "cell_id",
#     shapes_key: str = "cell_boundaries",
#     radius_factor: float = 2.0,
# ) -> dict:
#     """
#     Find neighboring cells based on centroid distance.

#     A cell `j` is considered a neighbor of focal cell `i` if:

#         distance(centroid_i, centroid_j) <= radius_i * radius_factor

#     where `radius_i` is the equivalent radius of cell `i`, computed from its area.

#     Parameters
#     ----------
#     sdata
#         SpatialData object.
#     tables_key : str, default="table"
#         Key in `sdata.tables` for the cell table.
#     tables_cell_id_key : str, default="cell_id"
#         Column in `sdata.tables[tables_key].obs` containing cell ids.
#     shapes_key : str, default="cell_boundaries"
#         Key in `sdata.shapes` containing cell polygons.
#     radius_factor : float, default=2.0
#         Distance threshold as a multiple of the focal cell's equivalent radius.

#     Returns
#     -------
#     dict
#         Mapping: focal cell id -> list of neighboring cell ids.
#     """
#     if radius_factor < 0:
#         raise ValueError("`radius_factor` must be >= 0.")

#     cells_gdf = sdata.shapes[shapes_key].copy()
#     ad = sdata.tables[tables_key]

#     # ensure unique, valid cell IDs
#     cell_ids = pd.Index(ad.obs[tables_cell_id_key]).unique()
#     cell_ids = cell_ids[cell_ids.isin(cells_gdf.index)]
#     cells_gdf = cells_gdf.loc[cell_ids]

#     # compute centroids
#     centroids = cells_gdf.geometry.centroid
#     coords = np.c_[centroids.x.to_numpy(), centroids.y.to_numpy()]

#     # equivalent radii
#     areas = cells_gdf.geometry.area.clip(lower=1e-6)
#     radii = np.sqrt(areas.to_numpy() / np.pi)

#     # KD-tree for fast neighbor lookup
#     tree = cKDTree(coords)

#     neighbors = {}

#     for i, cid in enumerate(cell_ids):
#         r = radii[i] * radius_factor
#         idx = tree.query_ball_point(coords[i], r)

#         # exclude self
#         nbrs = [cell_ids[j] for j in idx if j != i]
#         neighbors[cid] = nbrs

#     return neighbors
def _find_neighbors_by_distance(
    sdata,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    radius_factor: float = 1.0,
) -> dict:
    """
    Find neighboring cells based on distance to the focal cell boundary.

    A cell `j` is considered a neighbor of focal cell `i` if the minimum
    distance from the geometry of `j` to the boundary of `i` is less than or
    equal to:

        radius_factor * equivalent_radius(i)

    where equivalent radius is computed from the area of the focal cell.

    Parameters
    ----------
    sdata
        SpatialData object.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell table.
    tables_cell_id_key : str, default="cell_id"
        Column in `sdata.tables[tables_key].obs` containing cell ids.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` containing cell polygons.
    radius_factor : float, default=1.0
        Distance threshold expressed as a multiple of the focal cell's
        equivalent radius.

    Returns
    -------
    dict
        Mapping from focal cell id to a list of neighboring cell ids.
    """
    if radius_factor < 0:
        raise ValueError("`radius_factor` must be >= 0.")

    cells_gdf = sdata.shapes[shapes_key].copy()
    ad = sdata.tables[tables_key]

    cell_ids = pd.Index(ad.obs[tables_cell_id_key]).unique()
    cell_ids = cell_ids[cell_ids.isin(cells_gdf.index)]
    cells_gdf = cells_gdf.loc[cell_ids]

    areas = cells_gdf.geometry.area.clip(lower=1e-6)
    radii = np.sqrt(areas / np.pi)

    sindex = cells_gdf.sindex
    neighbors = {}

    for focal_id, focal_geom in cells_gdf.geometry.items():
        if focal_geom is None or focal_geom.is_empty or not focal_geom.is_valid:
            neighbors[focal_id] = []
            continue

        focal_boundary = focal_geom.boundary
        if focal_boundary is None or focal_boundary.is_empty:
            neighbors[focal_id] = []
            continue

        max_dist = float(radius_factor * radii.loc[focal_id])
        search_geom = focal_boundary.buffer(max_dist)

        candidate_idx = list(sindex.intersection(search_geom.bounds))
        candidates = cells_gdf.iloc[candidate_idx]

        nbrs = []
        for other_id, other_geom in candidates.geometry.items():
            if other_id == focal_id:
                continue
            if other_geom is None or other_geom.is_empty or not other_geom.is_valid:
                continue
            if other_geom.distance(focal_boundary) <= max_dist:
                nbrs.append(other_id)

        neighbors[focal_id] = nbrs

    return neighbors


def _build_transcript_table(
    sdata,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_gene_key: str = "feature_name",
    erosion_fraction: float = 0.2,
    neighborhood_radius_factor: float = 1.0,
):
    """
    Build a transcript-level representation for center, border, and neighborhood
    contributions.

    Each output row represents one transcript assigned to one focal cell and one
    region:
        0 = center
        1 = border
        2 = neighborhood

    Center and border rows are obtained by spatially joining transcript points
    to the corresponding focal-cell polygons and keeping only transcripts whose
    assigned cell id matches the focal cell.

    Neighborhood rows are obtained from transcripts assigned to neighboring
    cells, where neighbors are defined by distance to the focal cell boundary.

    Notes
    -----
    `cell_type_key` and `cell_type_query` are intentionally fixed to `None`
    here, so no cell-type filtering is applied.

    Parameters
    ----------
    sdata
        SpatialData object.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table containing cell ids.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` containing cell polygons.
    points_key : str, default="transcripts"
        Key in `sdata.points` containing transcript points.
    points_cell_id_key : str, default="cell_id"
        Column in the transcript table containing transcript-assigned cell ids.
    points_background_id : str, default="UNASSIGNED"
        Identifier used for background / unassigned transcripts.
    points_x_key, points_y_key : str
        Coordinate columns in the transcript table.
    points_gene_key : str, default="feature_name"
        Column containing gene names.
    erosion_fraction : float, default=0.2
        Fraction of equivalent cell radius used to erode each cell polygon to
        define the center region.
    neighborhood_radius_factor : float, default=1.0
        Neighbor distance threshold expressed as a multiple of the focal cell's
        equivalent radius.

    Returns
    -------
    tuple
        `(focal_ids, gene_codes, region_codes, genes)`, where each array has
        one entry per transcript-row contribution.
    """
    pts = _get_filtered_points_df(
        sdata=sdata,
        genes=None,
        cell_type_key=None,
        cell_type_query=None,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_gene_key=points_gene_key,
        points_background_id=points_background_id,
    )[[points_cell_id_key, points_gene_key, points_x_key, points_y_key]].copy()

    pts = pts.reset_index(drop=True)
    pts["point_id"] = np.arange(len(pts), dtype=np.int64)

    genes = pd.Index(sdata.tables[tables_key].var_names)
    gene_to_code = pd.Series(np.arange(len(genes), dtype=np.int32), index=genes)

    pts = pts[pts[points_gene_key].isin(genes)].copy()
    pts["gene_code"] = pts[points_gene_key].map(gene_to_code).astype(np.int32)

    pts_gdf = gpd.GeoDataFrame(
        pts,
        geometry=gpd.points_from_xy(pts[points_x_key], pts[points_y_key]),
        crs=sdata.shapes[shapes_key].crs,
    )

    center_gdf, border_gdf = _get_center_and_border_shapes(
        sdata=sdata,
        shapes_key=shapes_key,
        erosion_fraction_of_radius=erosion_fraction,
    )

    def _assign_region(region_gdf: gpd.GeoDataFrame, region_code: int) -> pd.DataFrame:
        region_id_key = region_gdf.index.name or "cell_id"
        rg = (
            region_gdf.reset_index()
            .rename(columns={region_id_key: "focal_id"})[["focal_id", "geometry"]]
        )

        out = gpd.sjoin(pts_gdf, rg, how="inner", predicate="within").drop(
            columns=["index_right"]
        )
        out = out[out["focal_id"] == out[points_cell_id_key]]

        return out[["focal_id", "gene_code"]].assign(
            region_code=np.int8(region_code)
        )

    center = _assign_region(center_gdf, 0)
    border = _assign_region(border_gdf, 1)

    neighbor_map = _find_neighbors_by_distance(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        radius_factor=neighborhood_radius_factor,
    )

    inverse_map = {}
    for focal_id, nbrs in neighbor_map.items():
        for nbr in nbrs:
            inverse_map.setdefault(nbr, []).append(focal_id)

    neigh = pts[[points_cell_id_key, "gene_code"]].copy()
    neigh["focal_id"] = neigh[points_cell_id_key].map(inverse_map)
    neigh = neigh.explode("focal_id").dropna(subset=["focal_id"])
    neigh = neigh[["focal_id", "gene_code"]].assign(region_code=np.int8(2))

    joint = pd.concat([center, border, neigh], ignore_index=True)

    return (
        joint["focal_id"].to_numpy(),
        joint["gene_code"].to_numpy(dtype=np.int32),
        joint["region_code"].to_numpy(dtype=np.int8),
        genes,
    )


def _normalize_to_proportions(
    x: np.ndarray,
    pseudocount: float = 0.0,
) -> np.ndarray:
    """
    Normalize a 1D nonnegative count vector to proportions.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional nonnegative count vector.
    pseudocount : float, default=0.0
        Value added to all entries before normalization.

    Returns
    -------
    np.ndarray
        Proportion vector with the same shape as `x`.
    """
    x = np.asarray(x, dtype=float).ravel()

    if np.any(x < 0):
        raise ValueError("Counts must be nonnegative.")
    if pseudocount < 0:
        raise ValueError("`pseudocount` must be >= 0.")

    x = x + pseudocount
    total = x.sum()

    if total <= 0:
        return np.zeros_like(x, dtype=float)

    return x / total


def _estimate_mixture_alpha_least_squares(
    p_border: np.ndarray,
    p_center: np.ndarray,
    p_neighborhood: np.ndarray,
) -> float:
    """
    Estimate the neighborhood mixture weight in proportion space.

    The model is:

        p_border ~ (1 - alpha) * p_center + alpha * p_neighborhood

    Alpha is estimated by least squares and clipped to [0, 1].

    Parameters
    ----------
    p_border : np.ndarray
        Border gene proportions.
    p_center : np.ndarray
        Center gene proportions.
    p_neighborhood : np.ndarray
        Neighborhood gene proportions.

    Returns
    -------
    float
        Estimated mixture weight in [0, 1].
    """
    d = p_neighborhood - p_center
    denom = float(np.dot(d, d))

    if np.isclose(denom, 0.0):
        return 0.0

    alpha = float(np.dot(p_border - p_center, d) / denom)
    return float(np.clip(alpha, 0.0, 1.0))


def _score_one_cell(
    x_center: np.ndarray,
    x_border: np.ndarray,
    x_neighborhood: np.ndarray,
    min_transcripts: int = 10,
    min_genes: int = 5,
    pseudocount: float = 0.5,
) -> float:
    """
    Compute the border admixture score for one cell.

    The border profile is modeled as a mixture of the center and neighborhood
    profiles in gene-proportion space:

        p_border ~ (1 - alpha) * p_center + alpha * p_neighborhood

    The returned score is the relative reduction in squared L2 error obtained
    by the fitted mixture compared with the center-only fit.

    Parameters
    ----------
    x_center, x_border, x_neighborhood : np.ndarray
        Gene count vectors for the center, border, and neighborhood regions.
    min_transcripts : int, default=10
        Minimum number of transcripts required in each region.
    min_genes : int, default=5
        Minimum number of genes present across the three regions combined.
    pseudocount : float, default=0.5
        Pseudocount used when converting counts to proportions.

    Returns
    -------
    float
        Border admixture score, or `np.nan` if the cell does not meet the
        minimum requirements or if the center-only error is zero.
    """
    x_center = np.rint(np.asarray(x_center)).astype(int)
    x_border = np.rint(np.asarray(x_border)).astype(int)
    x_neighborhood = np.rint(np.asarray(x_neighborhood)).astype(int)

    mask = (x_center + x_border + x_neighborhood) > 0
    x_center = x_center[mask]
    x_border = x_border[mask]
    x_neighborhood = x_neighborhood[mask]

    n_genes_used = int(mask.sum())
    n_center = int(x_center.sum())
    n_border = int(x_border.sum())
    n_neighborhood = int(x_neighborhood.sum())

    if (
        n_genes_used < min_genes
        or n_center < min_transcripts
        or n_border < min_transcripts
        or n_neighborhood < min_transcripts
    ):
        return np.nan

    p_center = _normalize_to_proportions(x_center, pseudocount=pseudocount)
    p_border = _normalize_to_proportions(x_border, pseudocount=pseudocount)
    p_neighborhood = _normalize_to_proportions(
        x_neighborhood, pseudocount=pseudocount
    )

    alpha_hat = _estimate_mixture_alpha_least_squares(
        p_border=p_border,
        p_center=p_center,
        p_neighborhood=p_neighborhood,
    )

    p_mix = (1.0 - alpha_hat) * p_center + alpha_hat * p_neighborhood

    err_center_only = float(np.sum((p_border - p_center) ** 2))
    err_mixture = float(np.sum((p_border - p_mix) ** 2))

    if np.isclose(err_center_only, 0.0):
        return np.nan

    return float((err_center_only - err_mixture) / err_center_only)


def _bootstrap_one_cell(
    gene_codes: np.ndarray,
    region_codes: np.ndarray,
    n_genes: int,
    n_boot: int = 200,
    min_transcripts: int = 10,
    min_genes: int = 5,
    pseudocount: float = 0.5,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Bootstrap the border admixture score for one cell.

    Transcript rows are resampled with replacement, count vectors are rebuilt,
    and the per-cell score is recomputed on each bootstrap replicate.

    Parameters
    ----------
    gene_codes : np.ndarray
        Gene index for each transcript row.
    region_codes : np.ndarray
        Region label for each transcript row:
        0 = center, 1 = border, 2 = neighborhood.
    n_genes : int
        Total number of genes in the fixed gene universe.
    n_boot : int, default=200
        Number of bootstrap replicates.
    min_transcripts : int, default=10
        Minimum number of transcripts required in each region.
    min_genes : int, default=5
        Minimum number of genes required across the three regions combined.
    pseudocount : float, default=0.5
        Pseudocount used when converting counts to proportions.
    ci_level : float, default=0.95
        Percentile confidence interval level.
    rng : np.random.Generator | None, default=None
        Random number generator. If None, a new generator is created.

    Returns
    -------
    dict
        Dictionary with:
        - `border_admixture_score`
        - `border_admixture_score_ci_low`
        - `border_admixture_score_ci_high`
    """
    if len(gene_codes) == 0:
        return {
            "border_admixture_score": np.nan,
            "border_admixture_score_ci_low": np.nan,
            "border_admixture_score_ci_high": np.nan,
        }

    if rng is None:
        rng = np.random.default_rng()

    x_center, x_border, x_neighborhood = _build_count_vectors(
        gene_codes=gene_codes,
        region_codes=region_codes,
        n_genes=n_genes,
    )

    score = _score_one_cell(
        x_center=x_center,
        x_border=x_border,
        x_neighborhood=x_neighborhood,
        min_transcripts=min_transcripts,
        min_genes=min_genes,
        pseudocount=pseudocount,
    )

    n_rows = len(gene_codes)
    boot_scores = []

    for _ in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)

        xb_center, xb_border, xb_neighborhood = _build_count_vectors(
            gene_codes=gene_codes[idx],
            region_codes=region_codes[idx],
            n_genes=n_genes,
        )

        boot_score = _score_one_cell(
            x_center=xb_center,
            x_border=xb_border,
            x_neighborhood=xb_neighborhood,
            min_transcripts=min_transcripts,
            min_genes=min_genes,
            pseudocount=pseudocount,
        )

        if np.isfinite(boot_score):
            boot_scores.append(boot_score)

    boot_scores = np.asarray(boot_scores, dtype=float)

    if len(boot_scores) == 0:
        ci_low = np.nan
        ci_high = np.nan
    else:
        alpha = 1.0 - ci_level
        ci_low, ci_high = np.quantile(
            boot_scores,
            [alpha / 2, 1 - alpha / 2],
        )

    return {
        "border_admixture_score": (
            float(score) if np.isfinite(score) else np.nan
        ),
        "border_admixture_score_ci_low": (
            float(ci_low) if np.isfinite(ci_low) else np.nan
        ),
        "border_admixture_score_ci_high": (
            float(ci_high) if np.isfinite(ci_high) else np.nan
        ),
    }