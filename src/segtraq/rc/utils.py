import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from geopandas import GeoDataFrame
from pandas import Series
from rtree.index import Index
from scipy.spatial import cKDTree
from shapely.geometry.base import BaseGeometry

from ..utils import _is_background, _looks_like_counts


def _compute_iou(poly1: BaseGeometry, poly2: BaseGeometry) -> float:
    """Compute IoU between two shape polygons."""

    if not (poly1.is_valid and poly2.is_valid):  # TODO - make polygons valid later
        return np.nan
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    return inter_area / union_area if union_area > 0 else 0.0


def _norm_log_df(df: pd.DataFrame, scale: float = 1e4) -> pd.DataFrame:
    # row-wise library size normalization + log1p
    sums = df.sum(axis=1).replace(0, np.nan)
    df_norm = df.div(sums, axis=0) * scale
    return np.log1p(df_norm).fillna(0.0)


def _process_cell(
    cell_row: Series,
    shapes_cell_id_key: str | None,
    id_key: str | None,
    nucleus_shapes: GeoDataFrame,
    nucleus_shapes_cell_id_key: str | None,
    nuc_sindex: Index,
) -> dict[str | int, str | int, int | None | float]:
    """For one cell polygon compute the IoU with the best-matching nucleus."""
    cell_geom = cell_row.geometry

    cell_id = cell_row[shapes_cell_id_key] if shapes_cell_id_key is not None else cell_row.name

    # Get candidate nuclei bounding boxes that overlap this cell's bbox
    candidate_idx = list(nuc_sindex.intersection(cell_geom.bounds))

    if not candidate_idx:
        return {id_key: cell_row.name, "best_nuc_id": np.nan, "IoU": 0.0}

    candidates = nucleus_shapes.iloc[candidate_idx]

    best_iou: float = 0.0
    best_nuc_id = np.nan
    for _, nuc in candidates.iterrows():
        nuc_geom = nuc.geometry
        iou = _compute_iou(cell_geom, nuc_geom)
        if pd.notna(iou) and iou > best_iou:
            best_iou = iou
            if nucleus_shapes_cell_id_key is None:
                best_nuc_id = nuc.name
            else:
                best_nuc_id = nuc[nucleus_shapes_cell_id_key]

    return {id_key: cell_id, "best_nuc_id": best_nuc_id, "IoU": best_iou}


def _shapes_by_feature_df(
    sdata: sd.SpatialData,
    tables_cell_id_key: str = "cell_id",
    region_key: str = "nucleus_boundaries",
    region_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_gene_key: str = "feature_name",
) -> pd.DataFrame:
    """
    Aggregate feature counts per region (nucleus or other), converting transcripts to 2D if needed.

    Parameters
    ----------
        sdata : SpatialData
            A `SpatialData` object containing segmented and transcript-assigned spatial
            transcriptomics data (images, tables, points, shapes and optional labels).
        tables_cell_id_key : str, default="cell_id"
            Column in the cell table uniquely identifying each cell.
        region_key : str, default="nucleus_boundaries"
            Key in `sdata.shapes` for defining the regions to aggregate by.
        region_cell_id_key : str default="cell_id"
            Column linking polygons to cell IDs. If `None` is provided, the shape index is used as the cell ID.
        points_key : str, default="transcripts"
            Key in `sdata.points` for spot/transcript-level data.
        points_gene_key : str, default="feature_name"
            Column specifying the gene/feature name for each transcript/spot.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by shapes ID, columns = features (genes/proteins), values = counts.
    """

    # perform aggregation
    sdata2 = sdata.aggregate(
        values=points_key,
        by=region_key,
        value_key=points_gene_key,
        agg_func="count",
        deep_copy=False,
    )
    ad = sdata2.tables["table"]

    arr = ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X

    gdf = sdata.shapes[region_key].copy()

    if region_cell_id_key is not None:
        gdf = gdf.set_index(region_cell_id_key, drop=True)
    elif gdf.index.name is None:
        gdf.index.name = tables_cell_id_key

    df_out = pd.DataFrame(arr, index=gdf.index, columns=ad.var_names)
    return df_out


def _get_center_and_border_shapes(
    sdata: sd.SpatialData,
    shapes_key: str = "cell_boundaries",
    shapes_cell_id_key: str | None = "cell_id",
    tables_cell_id_key: str = "cell_id",
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
    shapes_cell_id_key : str, default="cell_id"
        Column name linking shapes to cell IDs.
        If `None`, the shape index is used as the cell ID.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
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

    if shapes_cell_id_key is not None:
        id_key = shapes_cell_id_key
        cells_gdf = cells_gdf.set_index(id_key, drop=True)
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

    if shapes_cell_id_key is None:
        center_gdf.set_index(id_key, drop=True, inplace=True)
        border_gdf.set_index(id_key, drop=True, inplace=True)

    return center_gdf[center_gdf.geometry.notna()], border_gdf[border_gdf.geometry.notna()]


def _assign_nuc_to_transcripts(
    sdata,
    tables_key: str = "table",
    nucleus_shapes_key: str = "nucleus_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
):
    """
    Assigns nucleus IDs to transcripts by performing a spatial join
    between transcript coordinates and nucleus polygons.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned spatial
        transcriptomics data (images, tables, points, shapes and optional labels).
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table. Gene names in
        `sdata.tables[tables_key].var.index` should match the gene field in
        `sdata.points[points_key]` (see `points_gene_key`).
    nucleus_shapes_key : str, default="nucleus_boundaries"
        Key in `sdata.shapes` for nucleus boundary polygons, if available.
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

    Returns
    -------
    tx : pandas.DataFrame
        A subset transcripts dataframe with nuclear assignments in
        column `nuc_id`.
    """
    nucs_gdf = sdata.shapes[nucleus_shapes_key].copy()
    nucs_gdf.index.name = "nuc_id"

    # Subset transcripts
    pts = sdata.points[points_key]
    cols = [points_cell_id_key, points_gene_key, points_x_key, points_y_key]
    pts = pts[cols]
    is_background = _is_background(pts[points_cell_id_key], points_background_id)
    pts = pts[~is_background]

    valid_features = pd.Index(
        sdata.tables[tables_key].var_names
    )  # TODO - this might break, if var.index and points_gene_key do not match!
    # e.g. one is Ensemble key and one is gene_key

    pts = pts.dropna(subset=[points_gene_key])
    pts = pts[pts[points_gene_key].isin(valid_features)]

    transcripts = pts.compute()
    transcripts = transcripts.reset_index(drop=True)
    transcripts[points_gene_key] = transcripts[points_gene_key].astype("category")

    pts_gdf = gpd.GeoDataFrame(
        transcripts,
        geometry=gpd.points_from_xy(transcripts[points_x_key], transcripts[points_y_key]),
        crs=nucs_gdf.crs,  # assume same CRS
    )

    tx_in_nuc = gpd.sjoin(
        pts_gdf[["geometry"]],
        nucs_gdf[["geometry"]],
        how="left",
        predicate="within",
    )[["nuc_id"]]

    # remove duplicate assignments
    tx_in_nuc = tx_in_nuc[["nuc_id"]].groupby(level=0, observed=True).first()

    tx_in_cell = transcripts[[points_gene_key, points_cell_id_key]]
    tx = tx_in_cell.join(tx_in_nuc, how="left")

    return tx


def _group_points_by_regions(
    sdata: sd.SpatialData,
    region_key: str,
    tables_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_cell_id_key: str = "cell_id",
    region_cell_id_key: str | None = "cell_id",
) -> gpd.GeoDataFrame:
    """
    Aggregate transcript counts per region (e.g., cell centers or cell borders)
    by annotating each transcript with the region polygon it falls into.

    The function converts transcript coordinates into a GeoDataFrame, performs a
    spatial join with the region polygons (e.g., centers, borders), and then
    counts only those transcripts whose assigned region ID matches their cell ID.
    This ensures compatibility with 3D-aware segmentation, where transcripts may
    share x/y coordinates but belong to different z-resolved cells.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned spatial
        transcriptomics data (images, tables, points, shapes and optional labels).
    region_key : str
        Key in `sdata.shapes` specifying which regions to use (e.g., `"cell_centers"`,
        `"cell_borders"`). Must contain a `geometry` column with polygons.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    points_key : str, default="transcripts"
        Key in `sdata.points` for spot/transcript-level data.
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript/spot.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    region_cell_id_key : str or None, default="cell_id"
        Column in `sdata.shapes[region_key]` mapping each region polygon to a cell ID.
        If `None`, the shape index is used as the cell ID.

    Returns
    -------
    df_region : pandas.DataFrame
        A gene-by-region count matrix where:
            - Rows (`index`) correspond to region IDs (cell IDs).
            - Columns correspond to gene names.
            - Values are transcript counts within each region.
        Only transcripts whose region ID matches their own cell ID are counted.
    """
    pts = sdata.points[points_key].compute()

    pts_gdf = gpd.GeoDataFrame(
        pts,
        geometry=gpd.points_from_xy(pts[points_x_key], pts[points_y_key]),
        crs=sdata.shapes[region_key].crs,  # assume same CRS
    )

    region_gdf = sdata.shapes[region_key].copy()

    if region_cell_id_key is not None:
        id_key = region_cell_id_key
        region_gdf = region_gdf.rename(columns={region_cell_id_key: "region_id"})
    elif region_gdf.index.name is not None:
        id_key = region_gdf.index.name
        region_gdf.index.name = "region_id"
        region_gdf.reset_index(inplace=True)
    else:
        id_key = tables_cell_id_key
        region_gdf.index.name = "region_id"
        region_gdf.reset_index(inplace=True)

    region_gdf = region_gdf[["region_id", "geometry"]]

    pts_gdf_region = gpd.sjoin(
        pts_gdf,
        region_gdf,
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])

    df_region = (
        pts_gdf_region.loc[
            pts_gdf_region["region_id"] == pts_gdf_region[points_cell_id_key], ["region_id", points_gene_key]
        ]
        .groupby(["region_id", points_gene_key], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    df_region.index.name = id_key

    return df_region


def _compute_ncvs_within_radius(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    shapes_cell_id_key: str | None = "cell_id",
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
    shapes_cell_id_key : str or None, default="cell_id"
        Column in the shapes GeoDataFrame linking polygons to cell IDs.
        If `None`, the shape index is used as the cell ID.
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


def _assign_transcripts_to_center_or_border(
    sdata,
    shapes_key: str = "cell_boundaries",
    shapes_cell_id_key: str | None = "cell_id",
    tables_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_cell_id_key: str = "cell_id",
    erosion_fraction_of_radius: float = 0.3,
):
    center_gdf, border_gdf = _get_center_and_border_shapes(
        sdata=sdata,
        shapes_key=shapes_key,
        shapes_cell_id_key=shapes_cell_id_key,
        tables_cell_id_key=tables_cell_id_key,
        erosion_fraction_of_radius=erosion_fraction_of_radius,
    )

    sdata.shapes["cell_centers"] = sd.models.ShapesModel.parse(center_gdf, transformations=None)
    sdata.shapes["cell_borders"] = sd.models.ShapesModel.parse(border_gdf, transformations=None)

    cell_shape_transformation = sd.transformations.get_transformation(sdata.shapes[shapes_key])
    sd.transformations.set_transformation(sdata.shapes["cell_centers"], cell_shape_transformation)
    sd.transformations.set_transformation(sdata.shapes["cell_borders"], cell_shape_transformation)

    expr_center = _group_points_by_regions(
        sdata=sdata,
        region_key="cell_centers",
        tables_cell_id_key=tables_cell_id_key,
        points_key=points_key,
        points_gene_key=points_gene_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_cell_id_key=points_cell_id_key,
        region_cell_id_key=shapes_cell_id_key,
    )

    expr_border = _group_points_by_regions(
        sdata=sdata,
        region_key="cell_borders",
        tables_cell_id_key=tables_cell_id_key,
        points_key=points_key,
        points_gene_key=points_gene_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_cell_id_key=points_cell_id_key,
        region_cell_id_key=shapes_cell_id_key,
    )

    return center_gdf, border_gdf, expr_center, expr_border


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
