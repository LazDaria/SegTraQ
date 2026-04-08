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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import chi2_contingency, fisher_exact
from scipy.special import gammaln

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

def _norm_log_vector(x: np.ndarray, scale: float = 1e4) -> np.ndarray:
    # Library-size normalize a 1D count vector and apply log1p.
    total = x.sum()
    if total == 0:
        return np.zeros_like(x, dtype=float)
    return np.log1p((x / total) * scale)

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
    cell_type_key: str,
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
    Compute neighborhood gene-count vectors as the summed expression of
    neighboring cells within a user-defined distance.

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
            ncv_arr[i, :] = expr_cells.values[idxs, :].sum(axis=0)

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

def _cosine_sim(x: np.ndarray, y: np.ndarray) -> float:
    """Return cosine similarity between two 1D vectors."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm == 0.0 or y_norm == 0.0:
        return np.nan

    return float(np.dot(x, y) / (x_norm * y_norm))

def _random_partition_counts(
    pooled_counts: np.ndarray,
    n_first: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly split pooled counts into two vectors of sizes `n_first` and
    `pooled_counts.sum() - n_first`, without replacement.
    """
    pooled_counts = np.asarray(pooled_counts)
    pooled_counts = np.rint(pooled_counts).astype(int)

    total = int(pooled_counts.sum())
    if not 0 <= n_first <= total:
        raise ValueError("`n_first` must be between 0 and pooled_counts.sum().")

    if total == 0:
        zeros = np.zeros_like(pooled_counts, dtype=int)
        return zeros, zeros

    # Draw gene counts for the first partition directly, without expanding
    # to transcript-level labels.
    first_counts = rng.multivariate_hypergeometric(pooled_counts, n_first)
    second_counts = pooled_counts - first_counts

    return first_counts, second_counts

def _simulate_partition_null_cosine(
    x_first_raw: np.ndarray,
    x_second_raw: np.ndarray,
    n_first: int,
    n_sim: int,
    scale: float,
    rng: np.random.Generator,
    q_low: float = 0.025,
    q_high: float = 0.975,
) -> dict:
    """
    Simulate cosine similarities under a random partition null for two count vectors.

    Returns summary statistics of the simulated null distribution, including
    mean, SD, lower/upper quantiles, and interval width.
    """
    pooled = x_first_raw + x_second_raw
    sims = np.empty(n_sim, dtype=float)

    for i in range(n_sim):
        sim_first_raw, sim_second_raw = _random_partition_counts(
            pooled_counts=pooled,
            n_first=n_first,
            rng=rng,
        )
        sim_first = _norm_log_vector(sim_first_raw, scale=scale)
        sim_second = _norm_log_vector(sim_second_raw, scale=scale)
        sims[i] = _cosine_sim(sim_first, sim_second)

    ql = float(np.quantile(sims, q_low))
    qh = float(np.quantile(sims, q_high))

    return {
        "null_mean": float(np.mean(sims)),
        "null_sd": float(np.std(sims, ddof=1)) if n_sim > 1 else 0.0,
        "null_q_low": ql,
        "null_q_high": qh,
        "null_interval_width": qh - ql,
    }

def _null_corrected_center_border_neighborhood_one_cell(
    x_center_raw: np.ndarray,
    x_border_raw: np.ndarray,
    x_neighborhood_raw: np.ndarray,
    min_transcripts: int = 10,
    min_genes: int = 5,
    n_sim: int = 200,
    scale: float = 1e4,
    random_state: int | None = None,
    q_low: float = 0.025,
    q_high: float = 0.975,
) -> dict:
    """
    Compute null-corrected center-border and border-neighborhood similarities
    for one cell, including empirical null quantiles.
    """
    rng = np.random.default_rng(random_state)

    # Work on a shared gene space so both similarities use the same features.
    x_center_raw = np.rint(np.asarray(x_center_raw)).astype(int)
    x_border_raw = np.rint(np.asarray(x_border_raw)).astype(int)
    x_neighborhood_raw = np.rint(np.asarray(x_neighborhood_raw)).astype(int)

    mask = (x_center_raw + x_border_raw + x_neighborhood_raw) > 0
    x_center_raw = x_center_raw[mask]
    x_border_raw = x_border_raw[mask]
    x_neighborhood_raw = x_neighborhood_raw[mask]

    n_genes_used = int(mask.sum())
    n_center = int(x_center_raw.sum())
    n_border = int(x_border_raw.sum())
    n_neighborhood = int(x_neighborhood_raw.sum())

    result = {
        "similarity_center_border": np.nan,
        "similarity_center_border_null_mean": np.nan,
        "similarity_center_border_null_sd": np.nan,
        "similarity_center_border_null_q_low": np.nan,
        "similarity_center_border_null_q_high": np.nan,
        "similarity_center_border_null_interval_width": np.nan,
        "similarity_center_border_residual": np.nan,
        "similarity_center_border_zscore": np.nan,

        "similarity_border_neighborhood": np.nan,
        "similarity_border_neighborhood_null_mean": np.nan,
        "similarity_border_neighborhood_null_sd": np.nan,
        "similarity_border_neighborhood_null_q_low": np.nan,
        "similarity_border_neighborhood_null_q_high": np.nan,
        "similarity_border_neighborhood_null_interval_width": np.nan,
        "similarity_border_neighborhood_residual": np.nan,
        "similarity_border_neighborhood_zscore": np.nan,

        "contamination_score": np.nan,

        "center_counts_used": n_center,
        "border_counts_used": n_border,
        "neighborhood_counts_used": n_neighborhood,
        "n_genes_used": n_genes_used,
    }

    if (
        n_genes_used < min_genes
        or n_center < min_transcripts
        or n_border < min_transcripts
        or n_neighborhood < min_transcripts
    ):
        return result

    x_center = _norm_log_vector(x_center_raw, scale=scale)
    x_border = _norm_log_vector(x_border_raw, scale=scale)
    x_neighborhood = _norm_log_vector(x_neighborhood_raw, scale=scale)

    sim_obs_cb = _cosine_sim(x_center, x_border)
    sim_obs_bn = _cosine_sim(x_border, x_neighborhood)

    null_cb = _simulate_partition_null_cosine(
        x_first_raw=x_center_raw,
        x_second_raw=x_border_raw,
        n_first=n_center,
        n_sim=n_sim,
        scale=scale,
        rng=rng,
        q_low=q_low,
        q_high=q_high,
    )
    residual_cb = float(sim_obs_cb - null_cb["null_mean"])
    zscore_cb = (
        np.nan
        if np.isclose(null_cb["null_sd"], 0.0)
        else float(residual_cb / null_cb["null_sd"])
    )

    null_bn = _simulate_partition_null_cosine(
        x_first_raw=x_border_raw,
        x_second_raw=x_neighborhood_raw,
        n_first=n_border,
        n_sim=n_sim,
        scale=scale,
        rng=rng,
        q_low=q_low,
        q_high=q_high,
    )
    residual_bn = float(sim_obs_bn - null_bn["null_mean"])
    zscore_bn = (
        np.nan
        if np.isclose(null_bn["null_sd"], 0.0)
        else float(residual_bn / null_bn["null_sd"])
    )

    contamination_score = (
        np.nan if np.isnan(residual_cb) or np.isnan(residual_bn)
        else float(residual_bn - residual_cb)
    )

    result.update(
        {
            "similarity_center_border": float(sim_obs_cb),
            "similarity_center_border_null_mean": null_cb["null_mean"],
            "similarity_center_border_null_sd": null_cb["null_sd"],
            "similarity_center_border_null_q_low": null_cb["null_q_low"],
            "similarity_center_border_null_q_high": null_cb["null_q_high"],
            "similarity_center_border_null_interval_width": null_cb["null_interval_width"],
            "similarity_center_border_residual": residual_cb,
            "similarity_center_border_zscore": zscore_cb,

            "similarity_border_neighborhood": float(sim_obs_bn),
            "similarity_border_neighborhood_null_mean": null_bn["null_mean"],
            "similarity_border_neighborhood_null_sd": null_bn["null_sd"],
            "similarity_border_neighborhood_null_q_low": null_bn["null_q_low"],
            "similarity_border_neighborhood_null_q_high": null_bn["null_q_high"],
            "similarity_border_neighborhood_null_interval_width": null_bn["null_interval_width"],
            "similarity_border_neighborhood_residual": residual_bn,
            "similarity_border_neighborhood_zscore": zscore_bn,

            "contamination_score": contamination_score,
        }
    )

    return result

def _chi2_center_border_one_cell(
    x_center_raw: np.ndarray,
    x_border_raw: np.ndarray,
    min_transcripts: int = 10,
    min_genes: int = 5,
) -> dict:
    """
    Compute a chi-square test of homogeneity between center and border
    transcript count vectors for one cell.

    Parameters
    ----------
    x_center_raw : np.ndarray
        Raw gene counts for the center region.
    x_border_raw : np.ndarray
        Raw gene counts for the border region.
    min_transcripts : int, default=10
        Minimum total transcript count required in both center and border.
    min_genes : int, default=5
        Minimum number of genes with nonzero total count across center+border.

    Returns
    -------
    dict
        Dictionary containing chi-square statistic, p-value, degrees of freedom,
        Cramér's V, and count summaries.
    """
    x_center_raw = np.rint(np.asarray(x_center_raw)).astype(int)
    x_border_raw = np.rint(np.asarray(x_border_raw)).astype(int)

    # Restrict to genes used by at least one of the two regions.
    mask = (x_center_raw + x_border_raw) > 0
    x_center_raw = x_center_raw[mask]
    x_border_raw = x_border_raw[mask]

    n_genes_used = int(mask.sum())
    n_center = int(x_center_raw.sum())
    n_border = int(x_border_raw.sum())

    result = {
        "chi2_center_border_stat": np.nan,
        "chi2_center_border_pvalue": np.nan,
        "chi2_center_border_dof": np.nan,
        "chi2_center_border_cramers_v": np.nan,
        "center_counts_used": n_center,
        "border_counts_used": n_border,
        "n_genes_used": n_genes_used,
    }

    if (
        n_genes_used < min_genes
        or n_center < min_transcripts
        or n_border < min_transcripts
    ):
        return result

    observed = np.vstack([x_center_raw, x_border_raw])

    # Remove any all-zero columns just in case.
    keep = observed.sum(axis=0) > 0
    observed = observed[:, keep]

    if observed.shape[1] < min_genes:
        return result

    chi2_stat, pvalue, dof, expected = chi2_contingency(
        observed,
        correction=False,
    )

    # Cramér's V for a 2 x k table
    n_total = observed.sum()
    r, k = observed.shape
    denom = n_total * min(r - 1, k - 1)
    cramers_v = np.sqrt(chi2_stat / denom) if denom > 0 else np.nan

    result.update(
        {
            "chi2_center_border_stat": float(chi2_stat),
            "chi2_center_border_pvalue": float(pvalue),
            "chi2_center_border_dof": int(dof),
            "chi2_center_border_cramers_v": float(cramers_v),
            "n_genes_used": int(observed.shape[1]),
        }
    )

    return result

def _log_choose(n: np.ndarray | int, k: np.ndarray | int) -> np.ndarray:
    """Compute log(binomial(n, k)) in a numerically stable way."""
    n = np.asarray(n)
    k = np.asarray(k)
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _log_multivariate_hypergeom_2xg(x_first: np.ndarray, col_totals: np.ndarray) -> float:
    """
    Log-probability of a 2 x G table under the fixed-margins null,
    parameterized by the first row counts and the column totals.

    Under the null, the first row counts follow a multivariate hypergeometric:
        P(X = x_first) = prod_j C(col_totals_j, x_first_j) / C(N, n_first)

    where:
        - col_totals_j are the fixed column totals
        - n_first = sum(x_first)
        - N = sum(col_totals)
    """
    x_first = np.rint(np.asarray(x_first)).astype(int)
    col_totals = np.rint(np.asarray(col_totals)).astype(int)

    n_first = int(x_first.sum())
    N = int(col_totals.sum())

    if np.any(x_first < 0) or np.any(x_first > col_totals):
        return -np.inf

    log_num = np.sum(_log_choose(col_totals, x_first))
    log_den = _log_choose(N, n_first)
    return float(log_num - log_den)


def _random_2xg_table_fixed_margins(
    col_totals: np.ndarray,
    n_first: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a random 2 x G table with fixed row totals and column totals.

    The first row is sampled from a multivariate hypergeometric distribution;
    the second row is determined by the column totals.
    """
    col_totals = np.rint(np.asarray(col_totals)).astype(int)
    first = rng.multivariate_hypergeometric(col_totals, n_first)
    second = col_totals - first
    return first, second


def _fisher_freeman_halton_center_border_one_cell(
    x_center_raw: np.ndarray,
    x_border_raw: np.ndarray,
    min_transcripts: int = 10,
    min_genes: int = 5,
    n_sim: int = 5000,
    random_state: int | None = None,
) -> dict:
    """
    Monte Carlo Fisher-Freeman-Halton exact test for a 2 x G center-border table.

    This conditions on:
        - total center count
        - total border count
        - total count of each gene

    and estimates a Fisher-style p-value as the fraction of sampled null tables
    whose probability is less than or equal to the observed table probability.

    Parameters
    ----------
    x_center_raw : np.ndarray
        Raw gene counts for the center region.
    x_border_raw : np.ndarray
        Raw gene counts for the border region.
    min_transcripts : int, default=10
        Minimum total transcript count required in both center and border.
    min_genes : int, default=5
        Minimum number of genes with nonzero total count across center+border.
    n_sim : int, default=5000
        Number of Monte Carlo null tables to sample.
    random_state : int or None, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary with Fisher-style Monte Carlo test results and summaries.
    """
    rng = np.random.default_rng(random_state)

    x_center_raw = np.rint(np.asarray(x_center_raw)).astype(int)
    x_border_raw = np.rint(np.asarray(x_border_raw)).astype(int)

    # Restrict to genes present in at least one of the two regions.
    mask = (x_center_raw + x_border_raw) > 0
    x_center_raw = x_center_raw[mask]
    x_border_raw = x_border_raw[mask]

    n_genes_used = int(mask.sum())
    n_center = int(x_center_raw.sum())
    n_border = int(x_border_raw.sum())
    col_totals = x_center_raw + x_border_raw

    result = {
        "fisher_center_border_logprob": np.nan,
        "fisher_center_border_pvalue": np.nan,
        "fisher_center_border_mc_nsim": int(n_sim),
        "center_counts_used": n_center,
        "border_counts_used": n_border,
        "n_genes_used": n_genes_used,
    }

    if (
        n_genes_used < min_genes
        or n_center < min_transcripts
        or n_border < min_transcripts
    ):
        return result

    # Observed table probability under fixed margins
    logp_obs = _log_multivariate_hypergeom_2xg(
        x_first=x_center_raw,
        col_totals=col_totals,
    )

    # Monte Carlo null: sample tables with same margins
    logp_null = np.empty(n_sim, dtype=float)
    for i in range(n_sim):
        sim_center, _ = _random_2xg_table_fixed_margins(
            col_totals=col_totals,
            n_first=n_center,
            rng=rng,
        )
        logp_null[i] = _log_multivariate_hypergeom_2xg(
            x_first=sim_center,
            col_totals=col_totals,
        )

    # Fisher-style two-sided ordering:
    # tables at least as unlikely as observed -> logp <= logp_obs
    pvalue = (1 + np.sum(logp_null <= logp_obs)) / (n_sim + 1)

    result.update(
        {
            "fisher_center_border_logprob": float(logp_obs),
            "fisher_center_border_pvalue": float(pvalue),
        }
    )

    return result

def _normalize_to_proportions(x: np.ndarray, pseudocount: float = 0.0) -> np.ndarray:
    """
    Normalize a 1D nonnegative count vector to proportions with optional smoothing.
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
    Estimate alpha in:
        p_border ~ (1 - alpha) * p_center + alpha * p_neighborhood

    using least squares in proportion space, then clip to [0, 1].
    """
    d = p_neighborhood - p_center
    denom = float(np.dot(d, d))

    if np.isclose(denom, 0.0):
        return 0.0

    alpha = float(np.dot(p_border - p_center, d) / denom)
    return float(np.clip(alpha, 0.0, 1.0))

def _mixture_fit_contamination_one_cell(
    x_center_raw: np.ndarray,
    x_border_raw: np.ndarray,
    x_neighborhood_raw: np.ndarray,
    min_transcripts: int = 10,
    min_genes: int = 5,
    pseudocount: float = 0.5,
) -> dict:
    """
    Fit a simple center-neighborhood mixture model to the border profile for one cell.

    The model is:
        p_border ~ (1 - alpha) * p_center + alpha * p_neighborhood

    where all p_* are normalized gene proportions.

    Returns
    -------
    dict
        Dictionary containing:
            - alpha_hat: estimated contamination fraction in [0, 1]
            - fit_center_only_l2: squared L2 error of border vs center
            - fit_mixture_l2: squared L2 error of border vs fitted mixture
            - fit_improvement_l2: center-only error minus mixture error
            - fit_improvement_fraction: relative improvement over center-only
            - center/border/neighborhood counts used
            - n_genes_used
    """
    x_center_raw = np.rint(np.asarray(x_center_raw)).astype(int)
    x_border_raw = np.rint(np.asarray(x_border_raw)).astype(int)
    x_neighborhood_raw = np.rint(np.asarray(x_neighborhood_raw)).astype(int)

    # Shared gene space: retain genes present in at least one of the three regions
    mask = (x_center_raw + x_border_raw + x_neighborhood_raw) > 0
    x_center_raw = x_center_raw[mask]
    x_border_raw = x_border_raw[mask]
    x_neighborhood_raw = x_neighborhood_raw[mask]

    n_genes_used = int(mask.sum())
    n_center = int(x_center_raw.sum())
    n_border = int(x_border_raw.sum())
    n_neighborhood = int(x_neighborhood_raw.sum())

    result = {
        "mixture_alpha_hat": np.nan,
        "mixture_fit_center_only_l2": np.nan,
        "mixture_fit_mixture_l2": np.nan,
        "mixture_fit_improvement_l2": np.nan,
        "mixture_fit_improvement_fraction": np.nan,
        "center_counts_used": n_center,
        "border_counts_used": n_border,
        "neighborhood_counts_used": n_neighborhood,
        "n_genes_used": n_genes_used,
    }

    if (
        n_genes_used < min_genes
        or n_center < min_transcripts
        or n_border < min_transcripts
        or n_neighborhood < min_transcripts
    ):
        return result

    p_center = _normalize_to_proportions(x_center_raw, pseudocount=pseudocount)
    p_border = _normalize_to_proportions(x_border_raw, pseudocount=pseudocount)
    p_neighborhood = _normalize_to_proportions(x_neighborhood_raw, pseudocount=pseudocount)

    alpha_hat = _estimate_mixture_alpha_least_squares(
        p_border=p_border,
        p_center=p_center,
        p_neighborhood=p_neighborhood,
    )

    p_mix = (1.0 - alpha_hat) * p_center + alpha_hat * p_neighborhood

    # Squared L2 fit error in proportion space
    err_center_only = float(np.sum((p_border - p_center) ** 2))
    err_mixture = float(np.sum((p_border - p_mix) ** 2))

    improvement_l2 = err_center_only - err_mixture
    improvement_fraction = (
        np.nan if np.isclose(err_center_only, 0.0) else float(improvement_l2 / err_center_only)
    )

    result.update(
        {
            "mixture_alpha_hat": float(alpha_hat),
            "mixture_fit_center_only_l2": err_center_only,
            "mixture_fit_mixture_l2": err_mixture,
            "mixture_fit_improvement_l2": improvement_l2,
            "mixture_fit_improvement_fraction": improvement_fraction,
        }
    )

    return result