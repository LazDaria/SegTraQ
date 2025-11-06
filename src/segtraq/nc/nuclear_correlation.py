import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from joblib import Parallel, delayed
from pandas import DataFrame
from scipy.stats import pearsonr
from tqdm import tqdm

from ..utils import merge_into_obs
from .utils import _nucleus_by_feature_df, _process_cell


def compute_cell_nuc_ious(
    sdata: sd.SpatialData,
    shapes_cell_id_key: str = "cell_id",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    nucleus_shapes_key: str = "nucleus_boundaries",
    n_jobs: int = -1,
    use_progress: bool = True,
    inplace: bool = True,
) -> DataFrame:
    """
    Compute per-cell IoU between cell and nucleus boundaries in a SpatialData object.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        Must contain cell and nuclear shapes.
    shapes_cell_id_key: str
        Column in `sdata.shapes[shapes_key] containing cell IDs.
    tables_cell_id_key: str
        Column in the cell table uniquely identifying each cell.
    shapes_key : str, optional
        The key in the `shapes` attribute of `sdata` that corresponds to cell boundaries.
    nucleus_shapes_key : str, optional
        The key in the `shapes` attribute of `sdata` that corresponds to nucleus boundaries.
    n_jobs : int, optional
        Number of parallel jobs. Default=-1 uses all CPUs.
    use_progress : bool, optional
        Whether to display a progress bar with tqdm.
    inplace : bool, optional
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pandas.DataFrame
        Columns: [cell_id, best_nuc_id, IoU]
    """

    # Get GeoDataFrames
    cell_boundaries = sdata.shapes[shapes_key]
    nuc_boundaries = sdata.shapes[nucleus_shapes_key]

    # Build spatial index once
    nuc_sindex = nuc_boundaries.sindex

    # Iterator for cells
    iterator = cell_boundaries.iterrows()
    if use_progress:
        iterator = tqdm(
            iterator,
            total=len(cell_boundaries),
            desc="Processing IoU between cells and nuclei",
        )

    # Parallel loop over cells
    results = Parallel(n_jobs=n_jobs, verbose=0, prefer="threads")(
        delayed(_process_cell)(cell_row, shapes_cell_id_key, nuc_boundaries, nuc_sindex) for _, cell_row in iterator
    )

    iou_df = pd.DataFrame(results)

    if inplace:
        merge_into_obs(sdata, "table", iou_df, tables_cell_id_key, "cell_id")

    return iou_df


def compute_cell_nuc_correlation(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_cell_id_key: str = "cell_id",
    metric: str = "pearson",
    points_key: str = "transcripts",
    nucleus_shapes_key: str = "nucleus_boundaries",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    shapes_key: str = "cell_boundaries",
    n_jobs_iou: int = -1,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    For each cell in the SpatialData table, identifies the nucleus with highest IoU
    and computes a correlation (e.g. Pearson) between the gene expression profiles
    of the cell and that nucleus.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        A SpatialData object containing:
            - `.shapes[shapes_key]` and `.shapes['nucleus_boundaries']`
            for polygon geometries,
            - `.tables[tables_key]` as an AnnData table.
    tables_key : str
        Key in `sdata.tables` pointing to the expression matrix.
    tables_cell_id_key : str
        Column in `sdata.tables[tables_key].obs containing cell IDs to match with `shapes_cell_id_key`.
    shapes_cell_id_key : str or None, optional, default="cell_id"
        Column in the cell-boundary shapes linking polygons to cell IDs.
        If `None`, the shape index is used as the cell ID.
    metric : str
        Correlation metric. Currently supports only `"pearson"`.
    points_key : str
        Name of transcripts `Points` element.
    nucleus_shapes_key : str
        Name of nucleus shape layer to aggregate by.
    points_gene_key : str
        Column in transcripts pointing to feature (e.g. gene/protein).
    points_x_key: str
        Column in transcripts pointing x coordinate.
    points_y_key: str
        Column in transcripts pointing y coordinate.
    shapes_key: str
        Name of cell shape layer used for IoU if not yet calculated.
    n_jobs_iou: int
        Number of jobs for computing IoU, if not yet calculated.
    inplace : bool, optional
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
            - `cell_id`: identifier of each cell,
            - `best_nuc_id`: matching nucleus ID with highest IoU (or None),
            - `corr_nc_cell`: Pearson correlation between the cell and its matched nucleus gene counts
            (NaN if no match).
    """

    df = sdata.tables[tables_key].obs.copy()
    if "best_nuc_id" not in df.columns:
        iou_df = compute_cell_nuc_ious(
            sdata,
            shapes_cell_id_key,
            tables_cell_id_key,
            shapes_key=shapes_key,
            nucleus_shapes_key=nucleus_shapes_key,
            n_jobs=n_jobs_iou,
        )
        df = df.merge(
            iou_df,
            right_on="cell_id",
            left_on=tables_cell_id_key,
            how="left",
        )

    arr = (
        sdata.tables[tables_key].X.toarray()
        if hasattr(sdata.tables[tables_key].X, "toarray")
        else sdata.tables[tables_key].X
    )
    expr_cells = pd.DataFrame(
        arr,
        index=sdata.tables[tables_key].obs[tables_cell_id_key],
        columns=sdata.tables[tables_key].var.index,
    )

    expr_nucleus_df = _nucleus_by_feature_df(
        sdata, points_key, nucleus_shapes_key, points_gene_key, points_x_key, points_y_key
    )

    common_genes = expr_nucleus_df.columns.intersection(expr_cells.columns)
    expr_nucleus = expr_nucleus_df[common_genes]
    expr_cells = expr_cells[common_genes]

    rows = []
    for _, row in df.iterrows():
        cid, nid = row.cell_id, row.best_nuc_id
        if pd.isna(nid):  # if no overlapping nucleus
            rows.append(
                {
                    "cell_id": cid,
                    "best_nuc_id": np.nan,
                    "IoU": row.IoU,
                    "corr_nc_cell": 0.0,
                }
            )
        else:
            x = expr_cells.loc[cid, :].to_numpy().ravel()
            y = expr_nucleus.loc[nid, :].to_numpy().ravel()
            if metric == "pearson":
                corr, _ = pearsonr(x, y)
            else:
                raise ValueError(f"Metric {metric} not supported")  # TODO
            rows.append(
                {
                    "cell_id": cid,
                    "best_nuc_id": nid,
                    "IoU": row.IoU,
                    "corr_nc_cell": corr,
                }
            )

    corr_df = pd.DataFrame(rows)

    if inplace:
        merge_into_obs(sdata, tables_key, corr_df, tables_cell_id_key, "cell_id")

    return corr_df


def compute_correlation_between_parts(
    sdata,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    nucleus_shapes_key: str = "nucleus_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    n_jobs: int = 1,  # joblib not strictly needed; most win is from vectorization
    inplace: bool = True,
):
    """
    Vectorized version: computes Pearson correlation between the cell∩best_nucleus
    ("intersection") and the rest of the cell ("remainder") using spatial joins.
    Returns DataFrame with columns ["cell_id", "best_nuc_id", "IoU", "correlation_parts"].

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing cells, nuclei, and transcript points.
    tables_key : str
        Key in `sdata.tables` pointing to the expression matrix.
    tables_cell_id_key : str
        Column in `sdata.tables[tables_key].obs containing cell IDs to match with `shapes_cell_id_key`.
    shapes_cell_id_key:
        Column in `sdata.shapes[shapes_key] containing cell IDs.
    shapes_key : str
        Key for cell boundaries in sdata.shapes.
    nucleus_shapes_key : str
        Key for nucleus boundaries in sdata.shapes.
    points_key : str
        Key for transcript points in sdata.points.
    points_cell_id_key: str
        Column in the points table linking each transcript/spot to a cell.
    points_gene_key : str
        Feature column in transcript points (e.g. gene name).
    points_x_key : str
        Column name for x coordinate.
    points_y_key : str
        Column name for y coordinate.
    n_jobs : int
        Number of parallel jobs for correlation computation.
    inplace : bool, optional
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["cell_id", "best_nuc_id", "correlation_parts"]
    """

    if "best_nuc_id" not in sdata.tables[tables_key].obs.columns:
        iou_df = compute_cell_nuc_ious(
            sdata, shapes_cell_id_key, tables_cell_id_key, shapes_key, nucleus_shapes_key, n_jobs=n_jobs
        )
    else:
        iou_df = sdata.tables[tables_key].obs[["cell_id", "best_nuc_id", "IoU"]].copy()

    best_nuc_map = iou_df.set_index("cell_id")["best_nuc_id"]

    cells_gdf: gpd.GeoDataFrame = sdata.shapes[shapes_key]
    nucs_gdf: gpd.GeoDataFrame = sdata.shapes[nucleus_shapes_key]
    transcripts_df = sdata.points[points_key].compute()

    # Choose a single CRS (cells' CRS), and reproject other layers if needed - TODO
    target_crs = nucs_gdf.crs
    # if nucs_gdf.crs != target_crs:
    #    nucs_gdf = nucs_gdf.to_crs(target_crs)
    # transcripts -> GeoDataFrame
    transcripts_gdf = gpd.GeoDataFrame(
        transcripts_df,
        geometry=gpd.points_from_xy(transcripts_df[points_x_key], transcripts_df[points_y_key]),
        crs=transcripts_df.attrs.get("crs", target_crs) or target_crs,
    )
    # if transcripts_gdf.crs != target_crs:
    #     transcripts_gdf = transcripts_gdf.to_crs(target_crs)

    if shapes_cell_id_key is None:
        shapes_cell_id_key_fixed = "cell_id"
        cells_gdf[shapes_cell_id_key_fixed] = cells_gdf.index
    else:
        shapes_cell_id_key_fixed = shapes_cell_id_key

    tx_in_cell = gpd.sjoin(
        transcripts_gdf[[points_gene_key, "geometry"]],
        cells_gdf[[shapes_cell_id_key_fixed, "geometry"]],
        how="inner",
        predicate="within",
    )

    nucs_gdf.index.name = "nuc_id"
    tx_in_nuc = gpd.sjoin(
        transcripts_gdf[["geometry"]],
        nucs_gdf[["geometry"]],
        how="left",
        predicate="within",
    )[["nuc_id"]]

    tx = tx_in_cell.join(tx_in_nuc, how="left")

    tx["best_nuc_id"] = tx[points_cell_id_key].map(best_nuc_map)
    tx["in_intersection"] = (tx["nuc_id"].notna()) & (tx["nuc_id"] == tx["best_nuc_id"])
    tx["part"] = np.where(tx["in_intersection"], "intersection", "remainder")

    valid_features = pd.Index(sdata.tables[tables_key].var_names)
    tx = tx.dropna(subset=[points_gene_key])
    tx = tx[tx[points_gene_key].isin(valid_features)]
    tx[points_gene_key] = tx[points_gene_key].cat.remove_unused_categories()

    counts = tx.groupby([points_cell_id_key, "part", points_gene_key]).size().rename("count").reset_index()

    mat = counts.pivot_table(
        index=[points_cell_id_key, points_gene_key],
        columns="part",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )

    mat = mat[["intersection", "remainder"]]

    def _corr_two_cols(df_cell):
        x = df_cell["intersection"].to_numpy(dtype=float)
        y = df_cell["remainder"].to_numpy(dtype=float)
        if x.sum() == 0 or y.sum() == 0:
            return np.nan
        r, _ = pearsonr(x, y)
        return r

    corr_per_cell = mat.groupby(level=0, sort=False).apply(_corr_two_cols).rename("correlation_parts").to_frame()

    out = iou_df.set_index("cell_id")[["best_nuc_id", "IoU"]].join(corr_per_cell, how="left").reset_index()

    if inplace:
        merge_into_obs(sdata, tables_key, out, tables_cell_id_key, "cell_id")

    return out
