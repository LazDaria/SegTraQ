import collections
import copy
import math
import warnings
from collections.abc import Callable
from importlib.metadata import version

import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
import xarray as xr
from anndata import AnnData
from joblib import Parallel, delayed
from packaging import version as pkg_version
from rasterio.features import shapes
from scipy import sparse
from scipy.spatial.distance import cdist
from shapely.affinity import affine_transform, translate
from shapely.geometry import shape
from sklearn.metrics import roc_auc_score
from spatialdata.transformations import (
    get_transformation,
    get_transformation_between_coordinate_systems,
    set_transformation,
)

from .bl import baseline as bl


def _to_ndarray(x) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _looks_like_counts(x, n: int = 1000, tol: float = 1e-8) -> bool:
    """Quickly check if data looks like non-negative integer counts."""
    if sparse.issparse(x):
        # Nonzero entries only (zeros are fine for count check)
        arr = x.data
    elif hasattr(x, "values") and not isinstance(x, np.ndarray):
        arr = np.asarray(x.values).ravel()
    else:
        arr = np.asarray(x).ravel()

    if arr.size == 0:
        return False
    if np.issubdtype(arr.dtype, np.integer):
        return True

    samp = arr if arr.size <= n else np.random.choice(arr, n, replace=False)
    return np.all(samp >= 0) and np.allclose(samp, np.round(samp), atol=tol)


def _apply_overlap_filter(marker_dict: dict[str, list[str]], t, n_ct) -> dict[str, list[str]]:
    all_genes = [g for gl in marker_dict.values() for g in gl]
    if not all_genes:
        return {k: [] for k in marker_dict}
    counts = pd.Series(all_genes).value_counts()
    # drop genes appearing in >= t * n_types lists
    drop_genes = set(counts[counts >= (t * n_ct)].index)
    return {ct: [g for g in gl if g not in drop_genes] for ct, gl in marker_dict.items()}


def _score_one_list(
    X: np.ndarray,
    marker_idx: np.ndarray,
    all_markers_idx: np.ndarray | None = None,
    use_quantiles: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes precision, recall, F1 using upper-quantile rule for all cells simultaneously.

    Returns:
        precision: (n_cells,)
        recall:    (n_cells,)
        F1:        (n_cells,)
    """
    # Restrict scoring to a positive and negative markers
    all_markers_idx = np.asarray(all_markers_idx, dtype=int)
    X = X[:, all_markers_idx]
    n_cells, n_genes = X.shape

    # remap marker_idx into the all_markers_idx
    marker_set = set(np.asarray(marker_idx, dtype=int))
    marker_idx = np.array([k for k, g in enumerate(all_markers_idx) if g in marker_set], dtype=int)

    # No markers -> all metrics NaN for all cells
    if marker_idx.size == 0:
        return (np.full(n_cells, np.nan), np.full(n_cells, np.nan), np.full(n_cells, np.nan))

    # Boolean array marking actual positives
    actual = np.zeros(n_genes, dtype=bool)
    actual[marker_idx] = True

    frac = actual.mean()

    if use_quantiles:
        # Compute quantile threshold per cell
        thr = np.quantile(X, 1.0 - frac, axis=1)
        predicted = X > thr[:, None]
    else:
        predicted = X > 0

    actual_mat = np.broadcast_to(actual, (n_cells, n_genes))

    tp = (predicted & actual_mat).sum(axis=1)
    fp = (predicted & ~actual_mat).sum(axis=1)
    fn = (~predicted & actual_mat).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        F1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    return precision, recall, F1


def _score_negative_with_neighbors(
    X_dense: np.ndarray,
    cell_types: np.ndarray,
    markers: dict[str, dict[str, list[str]]],
    genes: np.ndarray,
    require_neighbor_expression: bool,
    neighbor_indices: list[np.ndarray],
    use_quantiles: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Neighborhood-aware negative marker scoring.

    For each cell i of type c:
      - Get its neighbors (indices neighbor_indices[i]).
      - Collect the cell types present in these neighbors.
      - Take negative markers of c: markers[c]["negative"].
      - Intersect with the union of positive markers of the neighbor types.
        -> cell-specific "relevant negatives".
      - Run `_score_one_list` on X_dense[i, :] with these markers to obtain
        precision, recall, F1 for negatives.

    Returns
    -------
    neg_precision, neg_recall, neg_F1 : (n_cells,) each
        Per-cell metrics for neighborhood-relevant negative markers.
    """
    n_cells, n_genes = X_dense.shape
    var_index = pd.Index(genes)

    # Precompute per-type sets for fast membership checks
    pos_sets: dict[str, set] = {}
    neg_sets: dict[str, set] = {}
    for ct, m in markers.items():
        pos_sets[ct] = set(m.get("positive", []))
        neg_sets[ct] = set(m.get("negative", []))

    neg_prec = np.full(n_cells, np.nan, dtype=float)
    neg_rec = np.full(n_cells, np.nan, dtype=float)
    neg_f1 = np.full(n_cells, np.nan, dtype=float)

    for i in range(n_cells):
        ct = cell_types[i]
        if pd.isna(ct) or ct not in markers:
            continue

        nbs = neighbor_indices[i]
        if nbs.size == 0:
            continue  # no neighborhood -> skip

        nb_cts = set(cell_types[nbs])
        if not nb_cts:
            continue

        neg_all = neg_sets.get(ct, set())
        if not neg_all:
            continue

        # union of positive markers from neighbor types
        nb_pos_union: set = set()
        for nb_ct in nb_cts:
            if nb_ct in pos_sets:
                nb_pos_union.update(pos_sets[nb_ct])

        # relevant negatives = negative markers of ct that are also
        # positive in at least one neighbor type
        rel_neg_genes = list(neg_all & nb_pos_union)
        if not rel_neg_genes:
            continue

        if require_neighbor_expression:
            keep_genes = []

            for g in rel_neg_genes:
                g_idx = var_index.get_loc(g)

                # check neighbors of types for which g is a positive marker
                for nb_ct in nb_cts:
                    if g not in pos_sets.get(nb_ct, set()):
                        continue

                    nb_mask = cell_types[nbs] == nb_ct
                    if nb_mask.any() and (X_dense[nbs[nb_mask], g_idx] > 0).any():
                        keep_genes.append(g)
                        break

            rel_neg_genes = keep_genes
            if not rel_neg_genes:
                continue

        # gene not present in spatial data
        neg_idx_i = var_index.get_indexer(rel_neg_genes)
        neg_idx_i = neg_idx_i[neg_idx_i >= 0]
        if neg_idx_i.size == 0:
            continue

        all_markers_idx = list(set(rel_neg_genes) | nb_pos_union)
        all_markers_idx_i = var_index.get_indexer(all_markers_idx)
        all_markers_idx_i = all_markers_idx_i[all_markers_idx_i >= 0]

        x_i = X_dense[i, :][None, :]  # (1, n_genes)
        n_prec_i, n_rec_i, n_f1_i = _score_one_list(
            x_i,
            neg_idx_i,
            all_markers_idx_i,
            use_quantiles=use_quantiles,
        )

        neg_prec[i] = n_prec_i[0]
        neg_rec[i] = n_rec_i[0]
        neg_f1[i] = n_f1_i[0]

    return neg_prec, neg_rec, neg_f1


def _assign_celltype_by_pearson(
    adata: AnnData, ref_mean_df: pd.DataFrame, q_ensemble_key: str = None, tables_cell_id_key: str = "cell_id"
) -> pd.DataFrame:
    """
    Assign cell types to cells in `adata` via Pearson correlation with reference means.

    Parameters
    ----------
    adata : AnnData
        Query dataset (log-normalized) with genes in `adata.var_names`.
    ref_mean_df : pd.DataFrame
        Reference matrix (cell_types x genes), log-normalized.
    query_ensemble_key: str or None, default="gene_ids"
        Column name in `self.sdata.tables[self.tables_key].var` that contains unique gene/ensemble IDs.
        If None, `self.sdata.tables[self.tables_key].var_names` will be used.
    tables_cell_id_key : str, default="cell_id"
        Column in the query cell table uniquely identifying each cell.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['cell_id', 'celltype', 'pearson_corr'].
    """
    genes = adata.var_names if q_ensemble_key is None else adata.var[q_ensemble_key]
    X_query = pd.DataFrame(
        _to_ndarray(adata.X),
        index=adata.obs[tables_cell_id_key],
        columns=genes,
    )

    common_genes = X_query.columns.intersection(ref_mean_df.columns)
    if len(common_genes) == 0:
        raise ValueError("No common genes found between query and reference.")

    X_query = X_query[common_genes]
    X_ref = ref_mean_df[common_genes]

    # correlation distance = 1 - Pearson correlation
    cor_mat = 1.0 - cdist(X_query.values, X_ref.values, metric="correlation")
    cor_df = pd.DataFrame(cor_mat, index=X_query.index, columns=X_ref.index)

    best_celltype = cor_df.idxmax(axis=1)
    best_score = cor_df.max(axis=1)

    return pd.DataFrame(
        {
            tables_cell_id_key: X_query.index,
            "transferred_cell_type": best_celltype.values,
            "pearson_score": best_score.values,
        }
    )


def run_label_transfer(
    sdata,
    adata_ref: AnnData,
    ref_cell_type: str,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_gene_key: str = "feature_name",
    tx_min: float = 10.0,
    tx_max: float = 2000.0,
    gn_min: float = 5.0,
    gn_max: float = np.inf,
    cell_type_key: str = "transferred_cell_type",
    ref_ensemble_key: str | None = None,
    query_ensemble_key: str | None = "gene_ids",
    inplace: bool = True,
) -> pd.DataFrame | None:
    """
    Transfer cell labels from a reference AnnData to `sdata.tables[tables_key]` by
    Pearson correlation to reference mean profiles.

    Parameters
    ----------
    sdata : SpatialData-like
        Container with `.tables[tables_key]` as AnnData, and points needed for QC if absent.
        `sdata.tables[tables_key].X` values are ideally normalized and log1p transformed.
        Otherwise transformation will be performed before running label transfer.
    adata_ref : AnnData
        Reference dataset (ideally normalized & log1p).
        Otherwise transformation will be performed before running label transfer.
    ref_cell_type : str
        Column in `adata_ref.obs` with reference cell types.
    tables_key : str
        Key of the AnnData table in `sdata.tables`.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    points_key : str, optional
        The key to access the transcript data within `sdata.points` (default is "transcripts").
    points_cell_id_key : str, optional
        The column name in the transcript data representing cell identifiers (default is "cell_id").
    points_gene_key : str, optional
        The column name in the transcript data representing gene names (default is "feature_name").
    tx_min, tx_max : float
        Min/max transcripts per cell for pre-filtering.
    gn_min, gn_max : float
        Min/max genes per cell for pre-filtering.
    cell_type_key : str
        Column name to store transferred labels in `.obs` when `inplace=True`.
    ref_ensemble_key: str or None, default=None
        Column name in `adata_ref.var` that contains unique gene/ensemble IDs.
        If None, `adata_ref.var_names` will be used.
    query_ensemble_key: str or None, default="gene_ids"
        Column name in `self.sdata.tables[self.tables_key].var` that contains unique gene/ensemble IDs.
        If None, `self.sdata.tables[self.tables_key].var_names` will be used.
    q_gene_key: str

    inplace : bool
        If True, writes labels into `sdata.tables[tables_key].obs` and returns None.
        If False, returns a DataFrame with ['cell_id', 'transferred_cell_type', 'pearson_score'].

    Returns
    -------
    None or pd.DataFrame
        None when `inplace=True`; otherwise a DataFrame of assignments.
    """

    if ref_cell_type not in adata_ref.obs.columns:
        raise KeyError(f"'{ref_cell_type}' not found in adata_ref.obs.")

    if _looks_like_counts(adata_ref.X):
        warnings.warn(
            "Reference adata_ref does not appear log-normalized."
            "Counts will be log1p-transformed before running label transfer."
            "Raw counts will be stored in `adata_ref.raw`.",
            RuntimeWarning,
            stacklevel=2,
        )
        adata_ref.raw = adata_ref.X.copy()
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)

    counts = _to_ndarray(adata_ref.X)
    celltypes = adata_ref.obs[ref_cell_type]
    genes = adata_ref.var_names if ref_ensemble_key is None else adata_ref.var[ref_ensemble_key].values
    counts_df = pd.DataFrame(counts, columns=genes)
    counts_df["celltype"] = celltypes.values
    ref_mean_df = counts_df.groupby("celltype").mean()

    tbl = sdata.tables[tables_key]
    # Ensure QC columns exist; compute if missing
    need_tx = "transcript_count" not in tbl.obs.columns
    need_gn = "gene_count" not in tbl.obs.columns

    if need_tx or need_gn:
        bl.transcripts_per_cell(
            sdata,
            tables_cell_id_key=tables_cell_id_key,
            points_key=points_key,
            points_cell_id_key=points_cell_id_key,
            tables_key=tables_key,
        )
        bl.genes_per_cell(
            sdata,
            tables_cell_id_key=tables_cell_id_key,
            points_key=points_key,
            points_cell_id_key=points_cell_id_key,
            points_gene_key=points_gene_key,
            tables_key=tables_key,
        )

    # QC filter
    qc_range = {"transcript_count": (tx_min, tx_max), "gene_count": (gn_min, gn_max)}
    mask = np.ones(tbl.n_obs, dtype=bool)
    for key, (low, high) in qc_range.items():
        if key not in tbl.obs.columns:
            raise KeyError(f"QC column '{key}' not found in table.obs.")
        mask &= (tbl.obs[key].to_numpy() >= low) & (tbl.obs[key].to_numpy() <= high)

    adata_q = tbl[mask]

    # Normalize & log1p (query)
    if _looks_like_counts(tbl.X):
        warnings.warn(
            "Spatialdata table appears to contain raw counts. "
            "Counts will be log1p-transformed before running label transfer."
            'Raw counts will be stored in `adata_q.layers["raw"]`.',
            RuntimeWarning,
            stacklevel=2,
        )
        adata_q.layers["raw"] = adata_q.X
        sc.pp.normalize_total(adata_q)
        sc.pp.log1p(adata_q)

    # Assign labels
    ct_corr = _assign_celltype_by_pearson(adata_q, ref_mean_df, query_ensemble_key, tables_cell_id_key)

    if inplace:
        # Write back only to the filtered subset cells
        out = ct_corr.rename(columns={"celltype": cell_type_key})
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )
        tbl.obs[cell_type_key] = tbl.obs[cell_type_key].astype("category")
        return None
    else:
        return out


def merge_into_obs(
    sdata, tables_key, df_to_merge: pd.DataFrame, tables_cell_id_key: str, df_cell_id_key: str, fillna_cols=None
):
    """
    Left-join df_to_merge into sdata.tables[tables_key].obs without resetting the index
    and without creating duplicate key columns.

    - Preserves obs index
    - Uses obs[tables_cell_id_key] as the join key unless df_cell_id_key already exists in obs
    - Drops overlapping columns on the right (or overwrites if overwrite=True)
    """

    obs = sdata.tables[tables_key].obs

    # Choose the column on the left to join on:
    # If the right's key already exists in obs, prefer that (avoids redundant columns)
    left_on_key = (
        df_cell_id_key if (df_cell_id_key == obs.index.name or df_cell_id_key in obs.columns) else tables_cell_id_key
    )

    # Build right indexed by the join key
    if df_to_merge.index.name != df_cell_id_key:
        right = df_to_merge.set_index(df_cell_id_key, drop=True)
    else:
        right = df_to_merge

    # Decide which columns from right to bring over
    right_cols = list(right.columns)
    overlapping_cols = [c for c in right_cols if c in obs.columns and c != df_cell_id_key]
    if overlapping_cols:
        obs = obs.drop(columns=overlapping_cols)

    # Perform a left join while preserving the left index.
    # Two cases: join using a left column (on=...) or directly on the index.
    if obs.index.name == left_on_key:
        # Index-on-index join (fast, preserves index)
        joined = obs.join(right, how="left")
    else:
        joined = obs.join(right, on=left_on_key, how="left")

    # Fill NAs if requested
    if fillna_cols:
        for c in fillna_cols:
            if c in joined.columns:
                joined[c] = joined[c].fillna(0)

    # Assign back (no intermediate index reset happened)
    sdata.tables[tables_key].obs = joined


def merge_into_var(sdata, tables_key, df_to_merge):
    var = sdata.tables[tables_key].var

    overlapping = [c for c in df_to_merge.columns if c in var.columns]

    if overlapping:
        var = var.drop(columns=overlapping)

    df = var.merge(df_to_merge, left_index=True, right_index=True, how="left")

    sdata.tables[tables_key].var = df


def _pairwise_auc(
    adata: AnnData,
    ctypes: pd.Categorical,
    ref_cell_type: str,
    ct_a: str,
    ct_b: str,
    max_fpr: float | None,
    auc_pos_thresh: float,
    min_cells_per_type: int,
) -> tuple[str, str, list[str], bool]:
    """
    Helper: compute per-gene AUC/pAUC for one pair (ct_a, ct_b)
    and return genes up in ct_a vs ct_b.
    """
    # Restrict to cells of ct_a and ct_b
    mask = ctypes.isin([ct_a, ct_b])
    if mask.sum() < 2 * min_cells_per_type:
        # too few cells total -> skip
        return (ct_a, ct_b, [], False)

    ad_pair = adata[mask]
    X_pair = ad_pair.X
    if hasattr(X_pair, "toarray"):
        X_pair = X_pair.toarray()
    else:
        X_pair = np.asarray(X_pair)  # (n_cells_pair, n_genes)

    genes = np.asarray(ad_pair.var_names)
    labels = (ad_pair.obs[ref_cell_type].values == ct_a).astype(int)
    if labels.sum() == 0 or labels.sum() == labels.size:
        # Only one class present -> skip
        return (ct_a, ct_b, [], False)

    # Precompute means per group to enforce directionality
    mask_a = labels == 1
    mask_b = labels == 0
    mean_a = X_pair[mask_a].mean(axis=0)
    mean_b = X_pair[mask_b].mean(axis=0)

    # Compute AUC/pAUC per gene (non-vectorized, one roc_auc_score per gene)
    aucs = np.zeros(genes.size, dtype=float)
    for j in range(genes.size):
        scores = X_pair[:, j]
        # If all scores identical, AUC is undefined -> treat as 0.5
        if np.all(scores == scores[0]):
            aucs[j] = 0.5
        else:
            aucs[j] = roc_auc_score(labels, scores, max_fpr=max_fpr)

    # Identify genes up in ct_a vs ct_b
    #   high AUC and higher mean in ct_a
    up_mask = (aucs >= auc_pos_thresh) & (mean_a > mean_b)
    pos_genes_a = genes[up_mask].tolist()

    return (ct_a, ct_b, pos_genes_a, True)


def _pairwise_de(
    adata: AnnData,
    ctypes: pd.Categorical,
    ref_cell_type: str,
    ct_a: str,
    ct_b: str,
    method: str,
    pval_adj_thresh: float,
    logfc_pos_thresh: float,
    min_cells_per_type: int,
) -> tuple[str, str, list[str], bool]:
    """
    Helper: run DE for one pair (ct_a, ct_b) and return genes up in ct_a.
    """
    mask = ctypes.isin([ct_a, ct_b])
    if mask.sum() < 2 * min_cells_per_type:
        # too few cells total, skip
        return (ct_a, ct_b, [], False)

    ad_pair = adata[mask].copy()

    # DE: ct_a vs ct_b
    sc.tl.rank_genes_groups(
        ad_pair,
        groupby=ref_cell_type,
        groups=[ct_a],
        reference=ct_b,
        method=method,
    )
    df = sc.get.rank_genes_groups_df(ad_pair, group=ct_a)

    pos_df = df[(df["pvals_adj"] < pval_adj_thresh) & (df["logfoldchanges"] > logfc_pos_thresh)]

    pos_genes_a = pos_df["names"].tolist()

    return (ct_a, ct_b, pos_genes_a, True)


def get_ref_markers(
    adata_ref: AnnData,
    ref_cell_type: str,
    mode: str = "de",
    max_fpr: float | None = None,
    auc_pos_thresh: float = 0.9,
    method: str = "wilcoxon",
    pval_adj_thresh: float = 0.05,
    logfc_pos_thresh: float = 1.0,
    vote_fraction_pos: float = 0.5,
    min_pos_frac: float = 0.1,
    max_neg_frac: float = 0.05,
    t_pos: float = 0.25,
    t_neg: float = 1.0,
    min_cells_per_type: int = 10,
    n_jobs: int = 1,
) -> dict[str, dict[str, list[str]]]:
    """
    Compute positive and negative markers per cell type using pairwise contrasts
    (AUC/pAUC or DE) followed by voting and a rarity-based definition of
    negative markers.

    Positive markers:
    -----------------
    For each cell type c, a gene g is considered a positive marker if it is
    "up in c" in at least ceil(vote_fraction_pos * M_c) of its valid pairwise
    comparisons (M_c). Additionally, g must be expressed (> 0) in at least
    min_pos_frac fraction of cells of type c in the reference dataset.

    Negative markers:
    -----------------
    For each ordered pair (a, b), take genes up in a vs b and consider them
    negative-marker candidates for b if they are expressed (> 0) in at most
    max_neg_frac fraction of cells of type b, and are not up in b vs any cell
    type (computed across all ordered contrasts).

    Overlap filtering:
    ------------------
    Overlap filtering is applied separately to positive and negative markers:
        - Positive lists: genes appearing in ≥ t_pos * n_types lists are dropped.
        - Negative lists: genes appearing in ≥ t_neg * n_types lists are dropped.

    Parameters
    ----------
    adata_ref : AnnData
        Reference single-cell dataset (cells x genes).
    ref_cell_type : str
        Column in `adata_ref.obs` containing cell type labels.
    mode : {"auc", "de"}, optional (default: "de")
        - "auc": compute markers using pairwise AUC/pAUC.
        - "de" : compute markers using pairwise DE.
    max_fpr : float or None, optional (default: None)
        (AUC mode only)
        If None, compute full AUC. If in (0, 1], compute standardized pAUC over
        [0, max_fpr] using sklearn's `roc_auc_score(max_fpr=max_fpr)`.
    auc_pos_thresh : float, optional (default: 0.9)
        (AUC mode only)
        Minimum AUC/pAUC for a gene to be considered "up in c_i vs c_j".
    method : str, optional (default: "wilcoxon")
        (DE mode only)
        DE method passed to `sc.tl.rank_genes_groups` ("wilcoxon", "t-test", "logreg", ...).
    pval_adj_thresh : float, optional (default: 0.05)
        (DE mode only)
        FDR (adjusted p-value) cutoff for positive markers.
    logfc_pos_thresh : float, optional (default: 1.0)
        (DE mode only)
        Minimum log fold-change for *positive* markers (c > d).
    vote_fraction_pos : float, optional (default: 0.5)
        Fraction of valid pairwise contrasts in which a gene must be "up in c"
        (AUC mode) / significantly up in c (DE mode) to be called a positive
        marker of c.
    min_pos_frac : float, optional (default: 0.1)
        Minimum fraction of cells of type c in which a gene must be expressed
        (counts > 0) in the reference dataset to be considered a *positive*
        marker of c.
    max_neg_frac : float, optional (default: 0.05)
        Maximum fraction of cells of type c in which a gene may be expressed
        (counts > 0) in the reference dataset to be considered a *negative*
        marker of c.
    t_pos : float, optional (default: 0.25)
        Overlap filter threshold for positive markers.
    t_neg : float, optional (default: 1.0)
        Overlap filter threshold for negative markers.
    min_cells_per_type : int, optional (default: 10)
        Minimum number of cells required per cell type to be included in pairwise
        computations.
    n_jobs : int, optional (default: 1)
        Number of parallel jobs for running pairwise computations.

    Returns
    -------
    dict
        {cell_type: {"positive": [genes], "negative": [genes]}}
    """
    adata = adata_ref.copy()

    # Normalize/log if this looks like raw counts
    if _looks_like_counts(adata.X):
        warnings.warn(
            "Reference adata_ref does not appear log-normalized. "
            "normalize_total + log1p will be applied to a copy for marker computation.",
            RuntimeWarning,
            stacklevel=2,
        )
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    adata.var_names_make_unique()

    ctypes = pd.Categorical(adata.obs[ref_cell_type])
    types = list(ctypes.categories)
    if len(types) < 2:
        raise ValueError("Need at least two cell types to compute markers.")

    # Cell counts per type -> filter rare types from pairwise contrasts
    cell_counts = ctypes.value_counts().to_dict()
    usable_types = [ct for ct in types if cell_counts.get(ct, 0) >= min_cells_per_type]
    n_types = len(usable_types)
    if n_types < 2:
        raise ValueError(
            f"Fewer than two cell types have at least {min_cells_per_type} cells; cannot perform pairwise contrasts."
        )

    # ordered pairs (a, b), a != b
    pairs = [(ct_a, ct_b) for ct_a in usable_types for ct_b in usable_types if ct_a != ct_b]

    var_names = np.asarray(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(var_names)}

    # Precompute fraction of cells with counts > 0 per type
    expr_frac: dict[str, np.ndarray] = {}
    X = adata.X
    for ct in usable_types:
        mask_ct = ctypes == ct
        X_ct = X[mask_ct]
        if sparse.issparse(X_ct):
            frac = X_ct.getnnz(axis=0) / X_ct.shape[0]
        else:
            frac = (X_ct > 0).mean(axis=0)

        expr_frac[ct] = np.asarray(frac).ravel()

    # Choose worker based on mode
    if mode == "auc":

        def worker(ct_a: str, ct_b: str):
            return _pairwise_auc(
                adata=adata,
                ctypes=ctypes,
                ref_cell_type=ref_cell_type,
                ct_a=ct_a,
                ct_b=ct_b,
                max_fpr=max_fpr,
                auc_pos_thresh=auc_pos_thresh,
                min_cells_per_type=min_cells_per_type,
            )

    elif mode == "de":

        def worker(ct_a: str, ct_b: str):
            return _pairwise_de(
                adata=adata,
                ctypes=ctypes,
                ref_cell_type=ref_cell_type,
                ct_a=ct_a,
                ct_b=ct_b,
                method=method,
                pval_adj_thresh=pval_adj_thresh,
                logfc_pos_thresh=logfc_pos_thresh,
                min_cells_per_type=min_cells_per_type,
            )

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'auc' or 'de'.")

    # Run over all pairs, possibly in parallel
    if n_jobs == 1:
        results = [worker(ct_a, ct_b) for ct_a, ct_b in pairs]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(worker)(ct_a, ct_b) for ct_a, ct_b in pairs)

    pair_df = pd.DataFrame(results, columns=["ct_a", "ct_b", "pos_genes_a", "ok"])
    pair_df = pair_df[pair_df["ok"]].reset_index(drop=True)

    up_by_pair: dict[tuple[str, str], list[str]] = {
        (row.ct_a, row.ct_b): list(row.pos_genes_a) for row in pair_df.itertuples(index=False)
    }

    # per-cell-type union of all "up" genes
    up_any: dict[str, set[str]] = {ct: set() for ct in usable_types}
    for row in pair_df.itertuples(index=False):
        up_any[row.ct_a].update(row.pos_genes_a)

    # -------------------------------------------------------------------------
    # Aggregate positives by voting; aggregate negatives by union (no vote filter)
    # -------------------------------------------------------------------------
    pos_votes = {ct: collections.Counter() for ct in usable_types}
    pair_counts_pos = {ct: 0 for ct in usable_types}  # valid (ct as "a")

    neg_sets = {ct: set() for ct in usable_types}  # accumulate negatives as a set (union)

    for ct_a, ct_b in pairs:
        up_ab = up_by_pair.get((ct_a, ct_b), set())
        if not up_ab:
            continue  # no genes passed in this contrast

        # positives: genes up in a vs b and common in a - vote for a
        frac_a = expr_frac[ct_a]
        for g in up_ab:
            idx = gene_to_idx.get(g)
            if idx is None:
                continue
            if frac_a[idx] >= min_pos_frac:
                pos_votes[ct_a][g] += 1
        pair_counts_pos[ct_a] += 1

        # negatives: genes up in a vs b, rare in b, AND not up in b vs any cell type
        frac_b = expr_frac[ct_b]
        up_b_any = up_any.get(ct_b, set())

        for g in up_ab:
            if g in up_b_any:
                continue
            idx = gene_to_idx.get(g)
            if idx is None:
                continue
            if frac_b[idx] <= max_neg_frac:
                neg_sets[ct_b].add(g)

    # ------------------------------------------------------------
    # Build positive marker lists using per-type voting thresholds
    # ------------------------------------------------------------
    pos_lists: dict[str, list[str]] = {}
    for ct in usable_types:
        M_c = pair_counts_pos.get(ct, 0)
        if M_c == 0:
            pos_lists[ct] = []
            continue

        min_pos_votes = max(1, int(np.ceil(vote_fraction_pos * M_c)))

        pos_genes = [g for g, k in pos_votes[ct].items() if (k >= min_pos_votes)]
        pos_genes = sorted(pos_genes, key=lambda g: pos_votes[ct][g], reverse=True)
        pos_lists[ct] = pos_genes

    # Overlap filter for positive markers
    pos_lists = _apply_overlap_filter(pos_lists, t=t_pos, n_ct=n_types)

    # ------------------------------------------------------------
    # Build negative marker lists
    # ------------------------------------------------------------
    # Keep negative markers only if they are positive markers of at least one other cell type
    pos_any_final: set[str] = set().union(*pos_lists.values()) if len(pos_lists) else set()
    for ct in usable_types:
        neg_sets[ct] = {g for g in neg_sets[ct] if g in pos_any_final}

    neg_lists: dict[str, list[str]] = {ct: sorted(list(neg_sets[ct])) for ct in usable_types}

    # Overlap filter for negative markers
    neg_lists = _apply_overlap_filter(neg_lists, t=t_neg, n_ct=n_types)

    markers: dict[str, dict[str, list[str]]] = {
        ct: {"positive": pos_lists.get(ct, []), "negative": neg_lists.get(ct, [])} for ct in usable_types
    }
    return markers


def _is_missing(x):
    """Return True for any kind of NA / NaN / None."""
    try:
        # Works for np.nan, float('nan'), pd.NA, pd.NaT, None
        return pd.isna(x) or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


def validate_spatialdata(
    sdata: sd.SpatialData,
    images_key: str | None = "morphology_focus",
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_area_volume_key: str | None = "cell_area",
    tables_centroid_x_key: str | None = "x_centroid",
    tables_centroid_y_key: str | None = "y_centroid",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_z_key: str | None = "z",
    points_gene_key: str = "feature_name",
    shapes_key: str | list[str] = "cell_boundaries",
    shapes_cell_id_key: str | None = "cell_id",
    nucleus_shapes_key: str | None = "nucleus_boundaries",
    nucleus_shapes_cell_id_key: str | None = "cell_id",
) -> bool:
    """
    Validates the integrity of a SpatialData object by checking the consistency of cell IDs
    across points, shapes, and tables.

    This function ensures that:
    - All points have corresponding shapes, and tables.
    - Cell IDs in points match those in shapes, and tables.
    - If shapes are present, they contain all cell IDs from the points.
    - If tables are present, they contain all cell IDs from the shapes.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object to validate.
    tables_key : str, optional
        Key for accessing tables in the SpatialData. Default is "table".
    tables_cell_id_key : str, optional
        Column name in the tables DataFrame (AnnData.obs) that contains cell IDs. Default is "cell_id".
    tables_centroid_x_key : str or None, optional, default="x_centroid"
        Column in the cell table with the x-coordinate of the cell centroid.
    tables_centroid_y_key : str or None, optional, default="y_centroid"
        Column in the cell table with the y-coordinate of the cell centroid.
    points_key : str, optional
        Key for accessing points (e.g., transcripts) in the SpatialData. Default is "transcripts".
    points_cell_id_key : str, optional
        Column name in the points DataFrame indicating cell assignments. Default is "cell_id".
    points_background_id : str, optional
        Identifier used for unassigned or background transcripts in the points DataFrame. Default is "UNASSIGNED".
    shapes_key : str or list of str, optional
        Key(s) for accessing shapes (e.g., cell boundaries) in the SpatialData. Default is "cell_boundaries".
        Can be a list if multiple shape layers are present.
    shapes_cell_id_key : str, optional
        Column name in the shapes DataFrame indicating cell IDs. Default is "cell_id".
        If None, the function assumes cell IDs are stored in the index.

    Raises
    ------
    TypeError
        If the input is not an instance of sd.SpatialData.
    ValueError
        If the SpatialData object does not contain points or if there are inconsistencies in cell IDs.

    Returns
    -------
    bool
        True if the SpatialData object passes all validation checks. Otherwise, an error or warning is raised.
    """
    if not isinstance(sdata, sd.SpatialData):
        raise TypeError("Input must be an instance of sd.SpatialData")

    # check if there is an image at the specified key
    if images_key is not None:
        assert images_key in sdata.images.keys(), (
            f"{images_key} not found in the image layer. "
            f"Available keys: {sdata.images.keys()}. "
            "You can set this with the images_key parameter (set to None if you do not have this)."
        )

    contains_points = len(sdata.points) > 0
    contains_shapes = len(sdata.shapes) > 0
    contains_tables = len(sdata.tables) > 0

    # check if there are points in the spatial data
    if not contains_points:
        raise ValueError("SpatialData object must contain points (transcripts)")

    # get the cell IDs from the points
    assert points_key in sdata.points, (
        f"SpatialData must contain points with key: {points_key}. "
        f"Available keys: {list(sdata.points.keys())}. "
        f"If you want to use a different key, set the points_key parameter."
    )
    points = sdata.points[points_key]

    # check gene column in points
    assert points_cell_id_key in points.columns, (
        f"Points DataFrame must contain column to identify cells: {points_cell_id_key}. "
        f"Available columns: {points.columns.tolist()}. "
        f"If you want to use a different column, set the points_cell_id_key parameter."
    )

    # check coordinate columns in points
    for coord_key, arg_name in [
        (points_x_key, "points_x_key"),
        (points_y_key, "points_y_key"),
    ]:
        assert coord_key in points.columns, (
            f"Points DataFrame must contain coordinate column '{coord_key}'. "
            f"Available columns: {points.columns.tolist()}. "
            f"You can set this with the '{arg_name}' argument."
        )

    if points_z_key is not None:
        assert points_z_key in points.columns, (
            f"Points DataFrame must contain z coordinate column '{points_z_key}'. "
            f"Available columns: {points.columns.tolist()}. "
            f"You can set this with the 'points_z_key' argument. Set to None if you do not have z coordinates."
        )

    # check gene key
    assert points_gene_key in points.columns, (
        f"Points DataFrame must contain gene feature column '{points_gene_key}'. "
        f"Available columns: {points.columns.tolist()}. "
        f"You can set this with the 'points_gene_key' argument."
    )

    if contains_tables:
        assert tables_key in sdata.tables, (
            f"Tables DataFrame must contain key: {tables_key}. "
            f"Available keys: {list(sdata.tables.keys())}. "
            f"If you want to use a different key, set the tables_key parameter."
        )
        table = sdata.tables[tables_key]
        if tables_area_volume_key is not None:
            assert tables_area_volume_key in table.obs.columns, (
                f"Tables DataFrame must contain area/volume column '{tables_area_volume_key}'. "
                f"Available columns: {table.obs.columns.tolist()}. "
                f"You can set this with the 'tables_area_volume_key' argument (set to None if you do not have this)."
            )

        if tables_centroid_x_key is not None:
            assert tables_centroid_x_key in table.obs.columns, (
                f"Tables DataFrame must contain x coordinate column '{tables_centroid_x_key}'. "
                f"Available columns: {table.obs.columns.tolist()}. "
                f"You can set this with the 'tables_centroid_x_key' argument (set to None if you do not have this)."
            )

        if tables_centroid_y_key is not None:
            assert tables_centroid_y_key in table.obs.columns, (
                f"Tables DataFrame must contain y coordinate column '{tables_centroid_y_key}'. "
                f"Available columns: {table.obs.columns.tolist()}. "
                f"You can set this with the 'tables_centroid_y_key' argument (set to None if you do not have this)."
            )

    # get unique cell IDs from points
    transcript_ids = set(points[points_cell_id_key].unique())
    shapes_cell_ids = set()

    # if there are shapes, ensure that there are no cell IDs in the points that are not in the shapes
    if contains_shapes:
        # we can have multiple shape keys (e. g. when using multiple layers in proseg), so we need to handle them here
        if isinstance(shapes_key, str):
            assert shapes_key in sdata.shapes, (
                f"Shapes DataFrame must contain key: {shapes_key}. "
                f"Available keys: {list(sdata.shapes.keys())}. "
                f"If you want to use a different key, set the shapes_key parameter."
            )
            shapes = sdata.shapes[shapes_key]
        elif isinstance(shapes_key, list):
            # if multiple shape keys are provided, we need to check each one
            shapes = pd.concat([sdata.shapes[key] for key in shapes_key], ignore_index=True)
        else:
            raise ValueError("shapes_key must be a string or a list of strings")

        # this part handles the case where cell IDs are stored in the index (as is the case in Xenium)
        shapes_cell_ids = set()
        if shapes_cell_id_key is None:
            shapes_cell_ids = set(shapes.index.tolist())
        else:
            assert shapes_cell_id_key in shapes.columns, (
                f"Shapes DataFrame must contain column: {shapes_cell_id_key}. "
                f"Available columns: {shapes.columns.tolist()}. "
                f"If you want to use a different column, set the shapes_cell_id_key parameter. "
                f"If you want to use the index as cell IDs, set shapes_cell_id_key=None."
            )
            shapes_cell_ids = set(shapes[shapes_cell_id_key])

        # ensuring that all cell IDs have the same dtype (either str or numeric)
        # taking a random ID from each set and comparing dtypes
        transcript_sample = next(iter(transcript_ids))
        shapes_sample = next(iter(shapes_cell_ids))

        def is_numeric(x):
            return isinstance(x, int | float | np.integer | np.floating)

        def is_string(x):
            return isinstance(x, str)

        if (is_numeric(transcript_sample) and is_numeric(shapes_sample)) or (
            is_string(transcript_sample) and is_string(shapes_sample)
        ):
            pass  # OK, both numeric or both string
        else:
            raise TypeError(
                f"Cell ID types between points and shapes are incompatible: "
                f"{type(transcript_sample)} (points) vs {type(shapes_sample)} (shapes). "
                f"Please ensure that cell IDs are all strings or all numeric."
            )

        # if the user provided a background ID, we want to ensure that it actually occurs
        if points_background_id is not None:
            assert points_background_id in transcript_ids, (
                f"points_background_id '{points_background_id}' not found among point cell IDs. "
                f"You can set this with the points_background_id parameter. "
                f"If you do not have a background ID, set this parameter to None."
            )

        # missing_in_polygons = { #TODO - after querying sdata objects, this breaks
        #     x
        #     for x in (transcript_ids - shapes_cell_ids - {points_background_id})
        #     if not _is_missing(x)  # also removing any NAs (no matter if from pandas, np, or None)
        # }
        # assert len(missing_in_polygons) == 0, (
        #     f"Missing {len(missing_in_polygons)} cell IDs from polygons: "
        #     f"{list(missing_in_polygons)[: min(5, len(missing_in_polygons))]}... "
        #     f"These cell IDs are present in the points, but not in the shapes. "
        #     f"If your missing cell ID is indicating an unassigned transcript, "
        #     f"you can set the points_background_id parameter."
        # )

        # if shapes and tables are present, ensure that the cell IDs match
        # checking that the adata and the polygons have the same cell IDs
        if contains_tables:
            assert tables_key in sdata.tables, (
                f"Tables DataFrame must contain key: {tables_key}. "
                f"Available keys: {list(sdata.tables.keys())}. "
                f"If you want to use a different key, set the tables_key parameter."
            )
            table = sdata.tables[tables_key]
            assert tables_cell_id_key in table.obs.columns, (
                f"Tables DataFrame must contain column: {tables_cell_id_key}. "
                f"Available columns: {table.obs.columns.tolist()}. "
                f"If you want to use a different column, set the tables_cell_id_key parameter."
            )

            assert "spatialdata_attrs" in table.uns, "Could not find 'spatialdata_attrs' in table.uns. "
            "You can set them like this: \n"
            "sdata.tables['table'].obs['region'] = 'cell_boundaries'\n"
            "sdata.set_table_annotates_spatialelement('table', region='cell_boundaries')"

            tables_cell_ids = set(table.obs[tables_cell_id_key].values)

            # checking that the dtype of the table cell IDs matches that of the shapes
            table_dtype = table.obs[tables_cell_id_key].dtype
            shapes_sample = next(iter(shapes_cell_ids))
            if is_numeric(shapes_sample):
                if not np.issubdtype(table_dtype, np.number):
                    raise TypeError(
                        f"Cell ID types between shapes and tables are incompatible: "
                        f"{type(shapes_sample)} (shapes) vs {table_dtype} (tables). "
                        f"Please ensure that cell IDs are all strings or all numeric."
                    )
            elif is_string(shapes_sample):
                if not np.issubdtype(table_dtype, np.object_) and not np.issubdtype(table_dtype, np.str_):
                    raise TypeError(
                        f"Cell ID types between shapes and tables are incompatible: "
                        f"{type(shapes_sample)} (shapes) vs {table_dtype} (tables). "
                        f"Please ensure that cell IDs are all strings or all numeric."
                    )

            # --- Ensure consistent types between shapes and tables ---
            # Ignore missing values (e.g. NaN, None) when checking type
            non_missing_shapes = [x for x in shapes_cell_ids if not _is_missing(x)]
            non_missing_tables = [x for x in tables_cell_ids if not _is_missing(x)]

            # Determine dominant type (str or numeric)
            shapes_has_str = any(isinstance(x, str) for x in non_missing_shapes)
            tables_has_str = any(isinstance(x, str) for x in non_missing_tables)

            # If one side contains strings, convert both sides to string
            if shapes_has_str or tables_has_str:
                shapes_cell_ids = {str(x) for x in shapes_cell_ids if not _is_missing(x)}
                tables_cell_ids = {str(x) for x in tables_cell_ids if not _is_missing(x)}
                points_background_id = str(points_background_id)
            else:
                # Ensure we drop any NAs (NaN, None, etc.) before comparison
                shapes_cell_ids = {x for x in shapes_cell_ids if not _is_missing(x)}
                tables_cell_ids = {x for x in tables_cell_ids if not _is_missing(x)}

            # --- Perform set comparisons ---
            missing_in_shapes = tables_cell_ids - shapes_cell_ids - {points_background_id}
            missing_in_tables = shapes_cell_ids - tables_cell_ids - {points_background_id}

            if len(missing_in_tables) != 0:
                warnings.warn(
                    f"Missing {len(missing_in_tables)} cell IDs in tables: "
                    f"{list(missing_in_tables)[: min(5, len(missing_in_tables))]}... "
                    "These cells are present in shapes, but not in tables. "
                    "This might lead to inconsistencies in the spatialdata object.",
                    stacklevel=2,
                )
            if len(missing_in_shapes) != 0:
                warnings.warn(
                    f"Missing {len(missing_in_shapes)} cell IDs in shapes: "
                    f"{list(missing_in_shapes)[: min(5, len(missing_in_shapes))]}... "
                    "These cells are present in tables, but not in shapes. "
                    "This might lead to inconsistencies in the spatialdata object.",
                    stacklevel=2,
                )

            # the checks above check the cell columns
            # however, spatialdata performs all joins on the indices, making it important that they match between
            # tables and shapes
            # we check if there is at least some overlap between the indices of the shapes and the tables
            # if not, we raise a warning
            shapes_index_ids = set(shapes.index.tolist())
            table_index_ids = set(table.obs.index.tolist())
            common_index_ids = shapes_index_ids & table_index_ids
            if len(common_index_ids) == 0:
                warnings.warn(
                    "The shapes and tables indices do not match. This will lead to errors when using spatialdata_plot. "
                    f"IDs in shapes index: {list(shapes_index_ids)[:5]}..., "
                    f"IDs in tables index: {list(table_index_ids)[:5]}...",
                    stacklevel=2,
                )

            # check that gene names in the table are compatible with those in the points
            genes_in_points = set(points[points_gene_key].unique())
            genes_in_table = set(table.var_names)
            common_genes = genes_in_points & genes_in_table
            if len(common_genes) == 0:
                raise ValueError(
                    "No common genes found between points and tables. "
                    "Please ensure that the gene names in both are compatible. "
                    f"Genes in points: {list(genes_in_points)[:5]}..., "
                    f"Genes in tables: {list(genes_in_table)[:5]}..."
                )

    # check for nucleus shapes
    if nucleus_shapes_key is not None:
        assert nucleus_shapes_key in sdata.shapes.keys(), (
            f"Nucleus shapes key '{nucleus_shapes_key}' not found in shapes. "
            f"Available keys: {list(sdata.shapes.keys())}. "
            f"You can set this with the 'nucleus_shapes_key' argument (set to None if you do not have this)."
        )

        if nucleus_shapes_cell_id_key is not None:
            nucleus_shapes = sdata.shapes[nucleus_shapes_key]
            assert nucleus_shapes_cell_id_key in nucleus_shapes.columns, (
                f"Nucleus shapes DataFrame must contain cell ID column '{nucleus_shapes_cell_id_key}'. "
                f"Available columns: {nucleus_shapes.columns.tolist()}. "
                "You can set this with the 'nucleus_shapes_cell_id_key' argument "
                "If you want to use the index as cell IDs, set nucleus_shapes_cell_id_key=None."
            )

    return True


def _process_image(
    sdata: sd.SpatialData,
    channel: str = "DAPI",
    images_key: str = "morphology_focus",
    images_data_key: str = "scale0/image",
    key_added: str = "nucleus_boundaries",
    return_values: bool = True,
):
    if key_added is not None:
        assert key_added not in sdata.labels.keys(), (
            f"Key {key_added} already exists in spatial data object. Please choose another key."
        )

    image = sdata.images[images_key]

    if isinstance(image, xr.DataTree):
        assert images_data_key is not None, (
            f"It looks like your image is stored as a DataTree. "
            f"Please provide a data_key to access the image data. "
            f"Available keys are: {list(image.keys())}."
        )
        assert images_data_key.split("/")[0] in image.keys(), (
            f"Data key {images_data_key} not found in the image data. Available keys: {list(image.keys())}"
        )

        image = image[images_data_key]

        assert isinstance(image, xr.DataArray), (
            f"The image data should be a DataArray. "
            f"Please provide a valid data key. "
            f"Available keys are: {[images_data_key + '/' + x for x in list(image.keys())]}."
        )

    try:
        image = image.sel(c=[channel])
    except KeyError as err:
        raise KeyError(
            f"Channel {[channel]} not found in the image data. Available channels: {list(image.c.values)}"
        ) from err

    if return_values:
        # returning a numpy array
        return image.values
    # returning an xarray object
    return image


def _cellpose(
    img: np.ndarray,
    diameter: float = None,
    channel_settings: list = None,
    num_iterations: int = 2000,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    batch_size: int = 8,
    gpu: bool = True,
    model_type: str = "cyto3",  # cellpose < 4.0
    pretrained_model: str = "cpsam",  # cellpose 4.0
    postprocess_func: Callable = lambda x: x,
    **kwargs,
):
    from cellpose import models

    cp_version = version("cellpose")

    # checking that the input is 2D or 3D
    if img.ndim not in [2, 3]:
        raise ValueError(f"Input image must be 2D or 3D, got {img.ndim}.")

    # if the input is 2D, we add a channel dimension
    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    # The cellpose API has changed in version 4.0, so we need to check the version

    if pkg_version.parse(cp_version).major < 4 and channel_settings != [0, 0]:
        assert channel_settings is not None, (
            "The argument channel_settings must be provided for Cellpose < 4.0. "
            "For independent segmentation of each channel, set it to [0, 0]. "
            "For joint segmentation, set it to [1, 2] or [2, 1]."
        )
        assert img.shape[0] == 2, (
            f"Joint segmentation requires exactly two channels. "
            f"You set channel_settings to {channel_settings}, "
            f"but provided {img.shape[0]} channels in the object."
        )
        model = models.Cellpose(gpu=gpu, model_type=model_type)
    else:
        # model_type is not used in cellpose 4.0
        model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

    all_masks = []
    # if the channels are [0, 0], independent segmentation is performed on all channels
    if channel_settings == [0, 0]:
        if img.shape[0] > 1:
            warnings.warn(
                "Performing independent segmentation on all markers. "
                "If you want to perform joint segmentation, "
                "please set the channel_settings argument appropriately.",
                RuntimeWarning,
                stacklevel=2,
            )
        for ch in range(img.shape[0]):
            # Build version-aware keyword arguments
            eval_kwargs = dict(
                diameter=diameter,
                niter=num_iterations,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                batch_size=batch_size,
                **kwargs,
            )

            if pkg_version.parse(cp_version).major < 4:
                eval_kwargs["channels"] = channel_settings

            # Get the image at the channel and run Cellpose
            output = model.eval(img[ch].squeeze(), **eval_kwargs)

            # Unpack outputs based on version
            if pkg_version.parse(cp_version).major >= 4:
                masks_pred, _, diams = output
            else:
                masks_pred, _, _, diams = output

            masks_pred = postprocess_func(masks_pred)
            all_masks.append(masks_pred)
    else:
        # if the channels are anything else, joint segmentation is attempted
        eval_kwargs = dict(
            diameter=diameter,
            niter=num_iterations,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            batch_size=batch_size,
            **kwargs,
        )

        if pkg_version.parse(cp_version).major < 4:
            eval_kwargs["channels"] = channel_settings

        output = model.eval(img.squeeze(), **eval_kwargs)

        # Unpack based on version
        if pkg_version.parse(cp_version).major >= 4:
            masks_pred, _, diams = output
        else:
            masks_pred, _, _, diams = output

        masks_pred = postprocess_func(masks_pred)
        all_masks.append(masks_pred)

    return np.array(all_masks), diams


def _labels_to_shapes(label_img: np.ndarray, simplify_tolerance: float | None = 0.5) -> gpd.GeoDataFrame:
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
        GeoDataFrame with columns ["cell_id", "geometry"]. Each row corresponds to one labeled region.
    """
    if label_img.ndim != 2:
        raise ValueError("Input label_img must be 2D.")

    mask = (label_img != 0).astype(np.uint8)
    geometries = []
    cell_ids = []

    # shapes() only accepts either of these dtypes: int16, int32, uint8, uint16, float32, float64, int8
    # if our labels are in uint32, and we have label_img.max() < 2_147_483_647, we need to convert to int32
    if label_img.dtype == np.uint32 and label_img.max() < np.iinfo(np.int32).max:
        label_img = label_img.astype(np.int32)

    for geom, value in shapes(label_img, mask=mask, connectivity=8):
        if value == 0:
            continue
        poly = shape(geom)
        # this turns disconnected polygons into MultiPolygons (which are valid)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_valid or poly.area == 0:
            continue
        if simplify_tolerance and simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance, preserve_topology=True)
        geometries.append(poly)
        cell_ids.append(int(value))

    gdf = gpd.GeoDataFrame({"cell_id": cell_ids, "geometry": geometries})
    # Merge multiple geometries per label into a single MultiPolygon
    gdf = gdf.dissolve(by="cell_id", as_index=True)
    # rasterio operates on the pixel centroids instead of the corners
    # to get sub-pixel accurate geometries, we need to shift the geometries by -0.5 in x and y
    gdf["geometry"] = gdf["geometry"].apply(lambda p: translate(p, xoff=-0.5, yoff=-0.5))

    return gdf


def add_nuc_shapes_via_cellpose(
    sdata: sd.SpatialData,
    channel: str = "DAPI",
    images_key: str = "morphology_focus",
    images_data_key: str = "scale0/image",
    shapes_key: str = "cell_boundaries",
    key_added: str = "nucleus_boundaries",
    inplace: bool = True,
    **kwargs,
):
    """
    This function runs the cellpose segmentation algorithm on the provided image data.
    It extracts the image data from the spatialdata object, applies the cellpose algorithm,
    and adds the segmentation masks to the spatialdata object.
    The segmentation masks are stored as polygons in the shapes attribute of the spatialdata object.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned spatial
        transcriptomics data (images, tables, points, shapes and optional labels).

    channel: str, default="DAPI"
        The channel(s) to be used for segmentation.

    images_key : str, default="morphology_focus"
        Key in `sdata.images` for a nuclear or morphology image (e.g., DAPI).
        Used for segmentation.

    images_data_key : str, default="scale0/image"
        Key for accessing data in `sdata.images` if they are stored as a DataTree.
        Consider using a higher scale (lower resolution) for segmentation to
        speed up computation and reduce memory usage during Cellpose.

    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons. Used to get transformations.

    key_added: str, default="nucleus_boundaries"
        The key under which the segmentation masks will be stored in the shapes attribute
        of the spatialdata object. Defaults to "nucleus_boundaries".

    inplace, bool, default=True
        Whether to modify the spatialdata object in place. Defaults to True.

    **kwargs: Additional keyword arguments to be passed to the cellpose algorithm.
    """

    if not inplace:
        sdata = copy.deepcopy(sdata)

    # assert that the format is correct and extract the image
    image = _process_image(
        sdata, channel=channel, images_key=images_key, key_added=key_added, images_data_key=images_data_key
    )

    # run cellpose
    segmentation_masks, _ = _cellpose(image, **kwargs)

    # convert labels to shapes
    nuc_shapes = _labels_to_shapes(segmentation_masks[0])

    # get transformations
    S = get_transformation(sdata.images[images_key][images_data_key]).to_affine_matrix(
        ("x", "y"), ("x", "y")
    )  # get scaling factors
    T = get_transformation_between_coordinate_systems(
        sdata, sdata.images[images_key], sdata.shapes[shapes_key]
    ).to_affine_matrix(("x", "y"), ("x", "y"))  # get affine transformation between image and shapes
    A = T @ S
    t_params = [A[0, 0], A[0, 1], A[1, 0], A[1, 1], A[0, 2], A[1, 2]]

    # apply affine transformation to nuclear shapes to have them in the same coordinate system as cell shapes
    nuc_shapes["geometry"] = nuc_shapes["geometry"].apply(lambda g: affine_transform(g, t_params))

    sdata.shapes[key_added] = sd.models.ShapesModel.parse(nuc_shapes, transformations=None)

    # set transformation for nucleus shapes to be the same as for cell shapes
    cell_shape_transformation = get_transformation(sdata.shapes[shapes_key])
    set_transformation(sdata.shapes[key_added], cell_shape_transformation)

    if not inplace:
        return sdata


def _is_background(series: pd.Series, background_id):
    """
    Checks if values in a Pandas Series match the background_id,
    handling None, np.nan, or pd.NA correctly.

    Args:
        series (pd.Series): The column data (e.g., 'cell_id').
        background_id: The value representing background (e.g., 'UNASSIGNED', 0, np.nan).

    Returns:
        pd.Series (bool): A boolean Series (True if background, False otherwise).
    """
    # Use pd.isna() to reliably check for None, np.nan, or pd.NA in background_id
    if pd.isna(background_id):
        # If the background ID is a null value, check for missing data in the Series
        is_background = series.isna()
    else:
        # Otherwise, perform a direct equality check
        is_background = series == background_id

    return is_background
