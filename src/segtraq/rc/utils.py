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
    if not candidate_idx:
        return {
            id_name: cell_id,
            "best_nuc_id": np.nan,
            "IoU": np.nan,
            "nucleus_fraction": np.nan,
        }

    candidates = nucleus_shapes.iloc[candidate_idx]

    cell_area = cell_geom.area if cell_geom.is_valid else np.nan

    best = {
        "score": -np.inf,
        "nuc_area": -np.inf,
        "inter_area": -np.inf,
        "nuc_id": np.nan,
        "iou": np.nan,
        "nuc_frac": np.nan,
    }

    for nuc_id, nuc in candidates.iterrows():
        nuc_geom = nuc.geometry
        if not (cell_geom.is_valid and nuc_geom.is_valid):
            continue

        nuc_area = nuc_geom.area
        if nuc_area <= 0 or cell_area <= 0:
            continue

        inter_area = _safe_intersection_area(cell_geom, nuc_geom)
        if np.isnan(inter_area) or inter_area <= min_intersection_area:
            continue

        iou = _compute_iou_from_areas(inter_area, cell_area, nuc_area)
        nuc_frac = _compute_nucleus_fraction(inter_area, nuc_area)

        score = iou if select_by == "iou" else nuc_frac

        # Compare with tie-breaks: score, then nuc_area, then inter_area, then nuc_id
        better = (
            (score > best["score"])
            or (np.isclose(score, best["score"]) and nuc_area > best["nuc_area"])
            or (
                np.isclose(score, best["score"])
                and np.isclose(nuc_area, best["nuc_area"])
                and inter_area > best["inter_area"]
            )
            or (
                np.isclose(score, best["score"])
                and np.isclose(nuc_area, best["nuc_area"])
                and np.isclose(inter_area, best["inter_area"])
                and nuc_id < best["nuc_id"]
            )
        )

        if better:
            best.update(
                score=score,
                nuc_area=nuc_area,
                inter_area=inter_area,
                nuc_id=nuc_id,
                iou=iou,
                nuc_frac=nuc_frac,
            )

    # If nothing survived filtering
    if best["score"] == -np.inf:
        return {
            id_name: cell_id,
            "best_nuc_id": np.nan,
            "IoU": np.nan,
            "nucleus_fraction": np.nan,
        }

    return {
        id_name: cell_id,
        "best_nuc_id": best["nuc_id"],
        "IoU": best["iou"],
        "nucleus_fraction": best["nuc_frac"],
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
    pts = sdata.points[points_key]
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
    transcripts = transcripts.reset_index(drop=False)  # keep old index as column, or create np.arange
    transcripts = transcripts.rename(columns={"index": "point_id"})

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

    ad = sdata.tables[tables_key]
    X = ad.X

    if _looks_like_counts(X):
        arr = X.toarray() if hasattr(X, "toarray") else X
    elif "raw" not in ad.layers:
        raise ValueError(
            f"'raw' layer does not exist in sdata.tables['{tables_key}'], "
            "and the main matrix does not look like counts."
        )
    else:
        raw = ad.layers["raw"]
        arr = raw.toarray() if hasattr(raw, "toarray") else raw

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
        idxs = tree.query_ball_point(coords[i], r=radii[i] * radius_factor)
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

    sdata.shapes["cell_centers"] = sd.models.ShapesModel.parse(center_gdf, transformations=None)
    sdata.shapes["cell_borders"] = sd.models.ShapesModel.parse(border_gdf, transformations=None)

    cell_shape_transformation = sd.transformations.get_transformation(sdata.shapes[shapes_key])
    sd.transformations.set_transformation(sdata.shapes["cell_centers"], cell_shape_transformation)
    sd.transformations.set_transformation(sdata.shapes["cell_borders"], cell_shape_transformation)

    _, expr_center = _join_points_regions(
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

    _, expr_border = _join_points_regions(
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

    return expr_center, expr_border


def _align_expression_dfs(dfs, sdata, tables_key: str = "table"):
    """Align multiple expression dataframes to have the same genes and cells."""
    # dfs is a dictionary of dataframes to align (key: name (e. g. 'expr_center'), value: dataframe)
    # ensure there are at least 2 dataframes
    if len(dfs) < 2:
        raise ValueError("At least two dataframes are required for alignment.")

    # Align dataframe columns to only keep common genes
    common_genes = list(dfs.values())[0].columns
    for i, (layer, other_df) in enumerate(dfs.items()):
        # skip first dataframe (we already have its columns)
        if i == 0:
            continue
        common_genes = common_genes.intersection(other_df.columns)
        if len(common_genes) == 0:
            raise ValueError(
                f"No common genes found when aligning layer {layer}. "
                f"Please ensure that your anndata object contains gene names in the var_names. "
                f"Previous gene names looked like: {list(list(dfs.values())[0].columns)[:5]}. "
                f"Gene names in layer {layer} look like: {list(other_df.columns)[:5]}."
            )

    # Only use gene`s transcripts and exclude control probes
    valid_genes = pd.Index(
        sdata.tables[tables_key].var_names
    )  # TODO - this might break, if var.index and points_gene_key do not match!
    # e.g. one is Ensemble key and one is gene_key
    common_genes = common_genes.intersection(valid_genes)

    dfs_aligned = {}
    for name, df in dfs.items():
        dfs_aligned[name] = df[common_genes]

    # Align dataframe rows- these might not match
    # expr_ncv computed based on table and expr_center/border based on shapes
    common_cells = dfs_aligned[list(dfs_aligned.keys())[0]].index
    for other_df in list(dfs_aligned.values())[1:]:
        common_cells = common_cells.intersection(other_df.index)

    for name, df in dfs_aligned.items():
        dfs_aligned[name] = df.loc[common_cells]

    return dfs_aligned
