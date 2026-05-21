import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from scipy.linalg import norm
from shapely.ops import unary_union
from sklearn.decomposition import PCA

from ..utils import _ensure_index, _is_background, merge_into_obs
from .utils import _correct_z_drift


def vertical_signal_integrity_per_cell(
    sdata,
    vsi_map: np.ndarray,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    inplace: bool = True,
):
    """
    Compute per-cell mean VSI by sampling a precomputed VSI map at transcript locations.

    This metric assumes `vsi_map` is defined on the same coordinate system and scaling
    as the transcript coordinates in `sdata.points[points_key]`, i.e. VSI values are
    indexed directly by integer x/y coordinates (after optional shift-to-origin).
    For each transcript, the VSI value is read from `vsi_map[y_int, x_int]` and then
    averaged across transcripts belonging to each cell.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing transcript-assigned spatial transcriptomics data.
    vsi_map : np.ndarray
        2D array of VSI values. Must be indexable as `vsi_map[y, x]`, where x/y correspond
        to transcript coordinates (after optional shift-to-origin).
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table. If `inplace=True`,
        results are merged into `sdata.tables[tables_key].obs`.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    points_key : str, default="transcripts"
        Key in `sdata.points` for transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell (background).
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript. Used to filter
        transcripts to features present in `sdata.tables[tables_key].var_names`.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript.
    inplace : bool, default=True
        Whether to add the results to `sdata.tables[tables_key].obs`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [tables_cell_id_key, "mean_vsi"]
    """
    if vsi_map.ndim != 2:
        raise ValueError(f"`vsi_map` must be a 2D array. Got shape {vsi_map.shape}.")

    # subset points and drop rows with missing cell identifiers, genes or coordinates
    pts = sdata.points[points_key]
    cols = [points_cell_id_key, points_gene_key, points_x_key, points_y_key]

    tx = pts[cols]
    tx = tx.dropna(subset=cols)

    # remove background transcripts
    is_bg = _is_background(tx[points_cell_id_key], points_background_id)
    tx = tx[~is_bg]

    # subset transcripts to genes present in the anndata object
    valid_features = pd.Index(sdata.tables[tables_key].var_names)
    tx = tx[tx[points_gene_key].isin(valid_features)]

    # convert to Pandas dataframe if Dask Array
    tx = tx.compute() if hasattr(tx, "compute") else tx
    tx = tx.reset_index(drop=True)

    # extract coordinates
    xs = tx[points_x_key].to_numpy(dtype=float)
    ys = tx[points_y_key].to_numpy(dtype=float)

    # shift coordinates to origin of coordinate system (0,0)
    # for ovrlpy to index correctly - requires positive indices
    x0 = float(np.min(xs))
    y0 = float(np.min(ys))
    xs = xs - x0
    ys = ys - y0

    # int floor for indexing
    xi = np.floor(xs).astype(int)
    yi = np.floor(ys).astype(int)

    # extract vsi values at coordinates
    vsi_vals = vsi_map[yi, xi].astype(float, copy=False)

    # cast into a dataframe
    df = pd.DataFrame(
        {
            tables_cell_id_key: tx[points_cell_id_key].to_numpy(),
            "vertical_signal_integrity": vsi_vals,
        }
    )

    # compute the cell wise mean of the vsi
    out = df.groupby(tables_cell_id_key, observed=True)["vertical_signal_integrity"].mean().reset_index()

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )

    return out


def _cell_by_gene_from_transcripts(tx_df: pd.DataFrame, cell_key: str, gene_key: str):
    return tx_df.groupby([cell_key, gene_key], observed=True).size().unstack(fill_value=0)


def _normalize(x: pd.DataFrame, normalization: str | None, scale: float = 1e4) -> pd.DataFrame:
    match normalization:
        case "log":
            x_sum = x.sum(axis=1).replace(0, np.nan)
            return np.log1p(x.div(x_sum, axis=0) * scale).fillna(0.0)
        case "raw" | None:
            return x
        case _:
            raise ValueError("Invalid `normalization`")


def similarity_top_bottom(
    sdata,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_z_key: str = "z",
    correct_z_drift: bool = True,
    max_points: int = 1_000_000,
    seed: int | None = 0,
    q: float = 0.30,
    normalization: str | None = None,
    scale: float = 1e4,
    min_genes: int = 5,
    min_transcripts: int = 10,
    n_pcs: int | None = 30,
    inplace: bool = True,
):
    """
    Compute cosine similarity between gene expression profiles of the bottom and top
    z-quantiles of transcripts within each cell.

    Optionally, a global z-drift correction is applied before computing within-cell
    quantiles (default: True). This is useful when raw z coordinates show tilt/warping
    across the field of view (e.g. slide not even in z).

    For each cell, transcripts are split into:
      - bottom part: z <= q-quantile within that cell
      - top part:    z >= (1-q)-quantile within that cell

    Gene counts normalized Analytic Pearson residuals (Lause et al. (2021)) for all
    counts together and work with the normalized residuals which are later taken apart

    Cells are filtered / set to NaN if either part is too sparse:
      - at least `min_transcripts` transcripts in BOTH bottom and top parts
      - at least `min_genes` genes with nonzero counts across (bottom OR top)

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing transcript-assigned spatial transcriptomics data.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    points_key : str, default="transcripts"
        Key in `sdata.points` for transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell (background).
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript.
    points_z_key : str, default="z"
        Column specifying the z coordinate / depth for each transcript.
    correct_z_drift : bool, default=True
        If True, correct global z-drift before computing within-cell z-quantiles.
        The corrected values are used only for defining top/bottom subsets.
    max_points : int, default=1_000_000
        Max. number of points used to fit the regression (random subsampling) in z drift correction.
    seed : int or None, default=0
        Random seed used for subsampling in z drift correction. If None, sampling is not reproducible.
    q : float, default=0.30
        Quantile defining bottom and top parts. bottom = q, top = 1-q.
    normalization: str, default="pearson"
        Normalization to be applied to the data. Either Pearson residuals ("pearson"),
        scaled log-transform ("log") or raw counts ("raw" or None).
    scale : float, default=1e4
        Scale for within-cell library size normalization (bottom+top).
    min_genes : int, default=5
        Minimum number of genes with nonzero counts in (bottom OR top) required to score a cell.
    min_transcripts : int, default=10
        Minimum number of transcripts required in EACH part (bottom and top) to score a cell.
    n_pcs : int | None, default=30
        Number of components to use for PCA. If None, no PCA will be performed.
    inplace : bool, default=True
        Whether to add the results to `sdata.tables[tables_key].obs`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [tables_cell_id_key, "cosine_sim_top_bottom_z"].
    """
    if not (0.0 < q < 0.5):
        raise ValueError(f"`q` must be in (0, 0.5). Got {q}.")

    # subset points and drop rows with missing cell identifiers, genes or coordinates
    pts = sdata.points[points_key]
    cols = [points_cell_id_key, points_gene_key, points_x_key, points_y_key, points_z_key]
    tx = pts[cols].dropna()

    # remove background transcripts
    is_bg = _is_background(tx[points_cell_id_key], points_background_id)
    tx = tx[~is_bg]

    # ensure genes match table var_names from the anndata object
    valid_features = sdata.tables[tables_key].var_names
    tx = tx[tx[points_gene_key].isin(valid_features)]

    # cast into pandas Dataframe if Dask Array
    tx = tx.compute() if hasattr(tx, "compute") else tx
    assert isinstance(tx, pd.DataFrame)
    tx = tx.reset_index(drop=True)

    # Optionally correct z-drift before defining top/bottom subsets
    if correct_z_drift:
        tx["_z_for_split"] = _correct_z_drift(
            tx=tx,
            points_x_key=points_x_key,
            points_y_key=points_y_key,
            points_z_key=points_z_key,
            max_points=max_points,
            seed=seed,
        )
    else:
        tx["_z_for_split"] = tx[points_z_key].to_numpy(dtype=float)

    # compute per-cell quantile cutoffs
    z_by_cell = tx.groupby(points_cell_id_key, observed=True)["_z_for_split"]
    tx["_z_bottom"] = z_by_cell.transform(lambda s: s.quantile(q))
    tx["_z_top"] = z_by_cell.transform(lambda s: s.quantile(1.0 - q))

    tx["_is_bottom"] = tx["_z_for_split"] <= tx["_z_bottom"]
    tx["_is_top"] = tx["_z_for_split"] >= tx["_z_top"]

    # counts per part
    counts_bottom = _cell_by_gene_from_transcripts(tx[tx["_is_bottom"]], points_cell_id_key, points_gene_key)
    counts_top = _cell_by_gene_from_transcripts(tx[tx["_is_top"]], points_cell_id_key, points_gene_key)

    # align top and bottom cells/genes
    common_cells = counts_bottom.index.intersection(counts_top.index)
    all_genes = tx[points_gene_key].unique()

    # counts of the common cells per bottom/top
    counts_bottom_raw = counts_bottom.loc[common_cells].reindex(columns=all_genes, fill_value=0)
    counts_top_raw = counts_top.loc[common_cells].reindex(columns=all_genes, fill_value=0)

    # total number of transcripts per bottom/top
    n_tx_bottom = counts_bottom_raw.sum(axis=1)
    n_tx_top = counts_top_raw.sum(axis=1)

    # filter cells by number of genes and min counts in top/bottom
    n_genes = ((counts_bottom_raw != 0) | (counts_top_raw != 0)).sum(axis=1)
    drop_cell = (n_tx_bottom < min_transcripts) | (n_tx_top < min_transcripts) | (n_genes < min_genes)

    counts_bottom_raw = counts_bottom_raw.loc[~drop_cell]
    counts_top_raw = counts_top_raw.loc[~drop_cell]
    common_cells = counts_bottom_raw.index

    top_norm = _normalize(counts_top_raw, normalization, scale)
    bottom_norm = _normalize(counts_bottom_raw, normalization, scale)

    # transform normalized counts into PCA space that is fit based on whole cells
    if n_pcs is not None:
        # TODO: probably tables["table"].X can be reused or similar
        counts_cell = _cell_by_gene_from_transcripts(tx, points_cell_id_key, points_gene_key)
        counts_cell = counts_cell.reindex(columns=all_genes, fill_value=0)
        cell_norm = _normalize(counts_cell, normalization, scale)
        pca = PCA(n_components=n_pcs, random_state=seed).fit(cell_norm)
        pca.set_output(transform="pandas")

        bottom_norm = pca.transform(bottom_norm)
        top_norm = pca.transform(top_norm)
        assert isinstance(bottom_norm, pd.DataFrame)
        assert isinstance(top_norm, pd.DataFrame)

    cosine_similarity = np.sum(top_norm * bottom_norm, axis=1) / (norm(top_norm, axis=1) * norm(bottom_norm, axis=1))
    out = cosine_similarity.to_frame("cosine_sim_top_bottom_z").rename_axis(tables_cell_id_key)

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )

    return out


def fraction_heterotypic_overlap(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    cell_type_key: str = "transferred_cell_type",
    shapes_key_list: list[str] = (
        "cell_boundaries_z0",
        "cell_boundaries_z1",
        "cell_boundaries_z2",
        "cell_boundaries_z3",
    ),
    shapes_cell_id_key: str = "cell_id",
    unknown_label: str = "Unknown",
    unknown_policy: str = "treat_as_label",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute cross-depth heterotypic overlap fraction per cell using one representative polygon per cell
    (chosen as the polygon with the largest area across z layers).

    For a representative polygon i (cell_id, z_layer) with geometry P_i and type t_i:

        overlap_area_i = Area( P_i ∩ Union_{j: z_j != z_i, id_j != id_i, t_j != t_i} P_j )
        overlap_fraction_i = overlap_area_i / Area(P_i)

    Candidates are restricted to bbox-overlapping polygons via a spatial index.

    Unknown/NA types:
      - unknown_policy="exclude": cells with NA/unknown type return NaN, and unknown-type
        candidates are excluded from overlap.
      - unknown_policy="treat_as_label": NA is replaced by `unknown_label` and treated as a
        real category.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing cell boundary polygons in multiple z layers and a
        cell table with transferred cell type labels.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    cell_type_key : str, default="transferred_cell_type"
        Column in the cell table containing cell-type labels (e.g. transferred from scRNA-seq).
    shapes_key_list : list[str] or tuple[str, ...]
        Keys in `sdata.shapes` for per-z-layer cell boundary polygons
        (e.g. ["cell_boundaries_z0", ..., "cell_boundaries_z3"]).
    shapes_cell_id_key : str, optional, default="cell_id"
        Index name of shapes GeoDataFrame linking polygons to cell IDs.
    unknown_label : str, default="Unknown"
        Label name to use when treating NA as a separate category (unknown_policy="treat_as_label").
    unknown_policy : str, default="exclude"
        How to handle Unknown/NA cell types:
          - "exclude": exclude polygons with NA/unknown types from comparisons. If the focal cell
            has NA/unknown type, its overlap fraction is set to NaN.
          - "treat_as_label": convert NA to `unknown_label` and treat it as a valid category.
    inplace : bool, default=True
        Whether to merge the aggregated per-cell result into `sdata.tables[tables_key].obs`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [tables_cell_id_key, "heterotypic_overlap_area", "heterotypic_overlap_fraction"].
    """
    if unknown_policy not in {"exclude", "treat_as_label"}:
        raise ValueError(f"unknown_policy must be one of {{'exclude','treat_as_label'}}. Got: {unknown_policy}")

    # extract meta data of cell observations
    obs = sdata.tables[tables_key].obs
    if cell_type_key not in obs.columns:
        raise KeyError(f"{cell_type_key!r} not found in sdata.tables[{tables_key!r}].obs")

    cell_type_map = obs.set_index(tables_cell_id_key)[cell_type_key].copy()

    gdfs = []
    # loop over the $z$-stacks
    for k, skey in enumerate(shapes_key_list):
        if skey not in sdata.shapes:
            raise KeyError(f"shapes key {skey!r} not found in sdata.shapes")
        # extract geopandas dataframe
        shapes = sdata.shapes[skey].copy()

        gdf = _ensure_index(shapes, shapes_key=skey, id_key=shapes_cell_id_key, id_key_name="shapes_cell_id_key")

        gdf["_cell_id"] = gdf.index
        # assign $z$-layer
        gdf["_z_layer"] = k
        gdf["_cell_type"] = gdf["_cell_id"].map(cell_type_map)

        if unknown_policy == "treat_as_label":
            gdf["_cell_type"] = gdf["_cell_type"].astype("object").where(gdf["_cell_type"].notna(), other=unknown_label)
        # subset geometry to be not NA and not empty
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()

        gdfs.append(gdf[["_cell_id", "_z_layer", "_cell_type", "geometry"]])

    gdf_all = pd.concat(gdfs, ignore_index=True)
    # making sure the concatenated table is a proper GeoPandas GeoDataFrame
    gdf_all = gpd.GeoDataFrame(gdf_all, geometry="geometry", crs=gdfs[0].crs if len(gdfs) else None)

    if unknown_policy == "exclude":
        gdf_all["_is_unknown"] = gdf_all["_cell_type"].isna() | (
            gdf_all["_cell_type"].astype("object") == unknown_label
        )
    else:
        gdf_all["_is_unknown"] = False

    gdf_all["_area"] = gdf_all.geometry.area.replace(0, np.nan)

    # pick representative polygon per cell: max area across layers
    rep_idx = gdf_all.groupby("_cell_id")["_area"].idxmax()
    reps = gdf_all.loc[rep_idx].copy()
    # get spatial index for cells in bbox around gdf polygons
    sindex = gdf_all.sindex

    # compute overlap fraction only for representative polygons
    out_rows = []
    for i, row in reps.iterrows():
        cid = row["_cell_id"]
        z_i = row["_z_layer"]
        t_i = row["_cell_type"]
        geom_i = row.geometry
        area_i = row["_area"]
        # store invalid cell areas
        if area_i is None or np.isnan(area_i) or area_i <= 0:
            out_rows.append((cid, np.nan, np.nan))
            continue

        if unknown_policy == "exclude":
            if bool(row["_is_unknown"]) or pd.isna(t_i):
                out_rows.append((cid, np.nan, np.nan))
                continue
        # get indices of the intersections of the spatial index with the
        # target cell
        cand_idx = list(sindex.intersection(geom_i.bounds))
        # exclude itself
        cand_idx = [j for j in cand_idx if j != i]
        # if empty add 0.0 - do here to avoid empty indexing
        if not cand_idx:
            out_rows.append((cid, 0.0, 0.0))
            continue

        cands = gdf_all.iloc[cand_idx]
        # only consider candidates that are not in the same $z$-layer
        cands = cands[cands["_z_layer"] != z_i]
        # only consider cells that are not the same cell_id (across $z$)
        cands = cands[cands["_cell_id"] != cid]

        if unknown_policy == "exclude":
            cands = cands[~cands["_is_unknown"]]
        # only consider cell types that are not of the same cell type
        cands = cands[cands["_cell_type"] != t_i]
        # if empty add 0.0
        if cands.empty:
            out_rows.append((cid, 0.0, 0.0))
            continue
        # compute area intersections between candidate cells and
        # target cell
        inter_geoms = []
        for geom_j in cands.geometry:
            inter = geom_i.intersection(geom_j)
            if (not inter.is_empty) and (inter.area > 0):
                inter_geoms.append(inter)

        if not inter_geoms:
            overlap_area = 0.0
        else:
            overlap_area = float(unary_union(inter_geoms).area)

        out_rows.append((cid, overlap_area, overlap_area / float(area_i)))

    per_cell_overlap_fraction = pd.DataFrame(
        out_rows, columns=[tables_cell_id_key, "heterotypic_overlap_area", "heterotypic_overlap_fraction"]
    )

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=per_cell_overlap_fraction,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )

    return per_cell_overlap_fraction
