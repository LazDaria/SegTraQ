import math
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from anndata import AnnData
from scipy import sparse
from scipy.spatial.distance import cdist

from .bl import baseline as bl


def _to_ndarray(x) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _looks_like_counts(x, n: int = 1000, tol: float = 1e-8) -> bool:
    """Quickly check if data looks like non-negative integer counts."""
    arr = x.data if hasattr(x, "data") else np.asarray(x).ravel()
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


def _score_one_list(expr: np.ndarray, marker_idx: np.ndarray, n_genes: int, use_quantiles: bool) -> tuple:
    """Precision, recall, F1 for one list using upper-quantile rule (CellSPA)."""
    if marker_idx.size == 0:
        return 0.0, 0.0, 0.0

    actual = np.zeros(n_genes, dtype=bool)
    actual[marker_idx] = True
    frac = actual.mean()

    if use_quantiles:
        thr = np.quantile(expr, 1.0 - frac)
        predicted = expr > thr
    else:
        predicted = expr > 0

    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, F1


def _assign_celltype_by_pearson(
    adata: AnnData, ref_mean_df: pd.DataFrame, q_ensemble_key: str = None, cell_id_key: str = "cell_id"
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
    cell_id_key: str
        Column name in tables DataFrame indicating cell IDs. Default is "cell_id".

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['cell_id', 'celltype', 'pearson_corr'].
    """
    genes = adata.var_names if q_ensemble_key is None else adata.var[q_ensemble_key]
    X_query = pd.DataFrame(
        _to_ndarray(adata.X),
        index=adata.obs[cell_id_key],
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
        {"cell_id": X_query.index, "transferred_cell_type": best_celltype.values, "pearson_score": best_score.values}
    )


def run_label_transfer(
    sdata,
    adata_ref: AnnData,
    ref_cell_type: str,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
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
            'Raw counts will be stored in `adata_ref.layers["raw"]`.',
            RuntimeWarning,
            stacklevel=2,
        )
        adata_ref.layers["raw"] = adata_ref.X
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
        bl.transcripts_per_cell(sdata)
        bl.genes_per_cell(sdata)

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
    ct_corr = _assign_celltype_by_pearson(adata_q, ref_mean_df, query_ensemble_key)

    if inplace:
        # Write back only to the filtered subset cells
        out = ct_corr.rename(columns={"celltype": cell_type_key})
        tbl.obs = tbl.obs.merge(out, how="left", left_on=tables_cell_id_key, right_on=tables_cell_id_key)
        tbl.obs[cell_type_key] = tbl.obs[cell_type_key].astype("category")
        return None
    else:
        return out


def merge_into_obs(sdata, tables_key, df_to_merge, table_cell_id_key, df_cell_id_key, fillna_cols=None):
    obs = sdata.tables[tables_key].obs

    # Drop overlapping columns, but keep the merge key
    overlapping = [
        c for c in df_to_merge.columns if c in obs.columns and c != table_cell_id_key and c != df_cell_id_key
    ]
    if overlapping:
        obs = obs.drop(columns=overlapping)

    # Merge
    df = obs.merge(df_to_merge, left_on=table_cell_id_key, right_on=df_cell_id_key, how="left")

    # Optionally fill numeric columns with zeros
    if fillna_cols:
        for c in fillna_cols:
            if c in df:
                df[c] = df[c].fillna(0)

    sdata.tables[tables_key].obs = df


def get_ref_markers(
    adata_ref: AnnData,
    ref_cell_type: str,
    q_pos: float = 0.95,
    q_neg: float = 0.10,
    t: float = 0.25,
) -> dict[str, dict[str, list[str]]]:
    """
    Translated and modified from https://github.com/SydneyBioX/CellSPA

    For each cell type c:
        w_g = mean(expression of gene g in cells of c) - mean(expression of g in all other cells)
        - positive markers(c): genes with w_g > quantile(w, q)
        - negative markers(c): genes with w_g < quantile(w, 1 - q)

    After building per-type lists, remove genes that occur in >= t * n_types lists
    (done separately for positives and negatives) to keep type-specific markers.

    Parameters
    ----------
    adata_ref : AnnData
        Reference single-cell dataset (cells x genes).
    ref_cell_type : str
        Column in `adata_ref.obs` containing cell type labels.
    q_pos : float, optional (default: 0.95)
        Upper quantile for positives.
    q_neg : float, optional (default: 0.10)
        Lower quantile for negatives.
    t : float, optional (default: 0.25)
        Overlap filter: drop genes that appear in >= t * n_types marker lists.

    Returns
    -------
    dict
        {cell_type: {"positive": [genes], "negative": [genes]}}
    """

    if _looks_like_counts(adata_ref.X):
        warnings.warn(
            "Reference adata_ref does not appear log-normalized."
            "Counts will be log1p-transformed before running label transfer.",
            RuntimeWarning,
            stacklevel=2,
        )
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)

    X = adata_ref.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    genes = np.asarray(adata_ref.var_names)
    ctypes = pd.Categorical(adata_ref.obs[ref_cell_type])
    types = list(ctypes.categories)
    n_types = len(types)
    if n_types < 2:
        raise ValueError("Need at least two cell types to compute differential markers.")

    # compute per-type mean expression (genes x types)
    means = {}
    for ct in types:
        mask = ctypes == ct
        if mask.sum() == 0:
            means[ct] = np.zeros(adata_ref.n_vars, dtype=float)
        else:
            means[ct] = X[mask].mean(axis=0)
    ref_exprs = pd.DataFrame(means, index=genes)

    # differential score w = mean_in_type - mean_in_others
    pos_lists: dict[str, list[str]] = {}
    neg_lists: dict[str, list[str]] = {}
    type_cols = ref_exprs.columns.to_list()

    for ct in type_cols:
        in_ct = ref_exprs[ct].to_numpy()
        others = ref_exprs.drop(columns=[ct]).mean(axis=1).to_numpy()
        w = in_ct - others

        # quantile cutoffs
        q_hi = np.quantile(w, q_pos)
        q_lo = np.quantile(w, q_neg)

        # positives = top-q
        pos_genes = ref_exprs.index[w > q_hi].tolist()
        # negatives = bottom-q
        neg_genes = ref_exprs.index[w < q_lo].tolist()

        pos_lists[ct] = pos_genes
        neg_lists[ct] = neg_genes

    # overlap filter (remove ubiquitous markers)
    pos_lists = _apply_overlap_filter(pos_lists, t=t, n_ct=n_types)
    neg_lists = _apply_overlap_filter(neg_lists, t=1, n_ct=n_types)

    markers = {ct: {"positive": pos_lists.get(ct, []), "negative": neg_lists.get(ct, [])} for ct in types}

    return markers


def get_mut_excl_markers(
    adata_ref,
    markers,
    ref_cell_type: str,
    pos_threshold: float = 0.20,
    neg_threshold: float = 0.05,
    max_codetect: float = 0.01,
    cell_types: tuple[str, str] | None = None,
) -> list[tuple[str, str]]:
    """
    Modified from https://github.com/dpeerlab/segger-analysis

    Finds mutually exclusive markers (presence-based specificity) between cell types.

    Optionally restricts computation to a specified pair of cell types.

    Parameters
    ----------
    adata_ref : AnnData
        Reference single-cell dataset (cells × genes).
    markers : dict
        Marker dictionary as returned by `find_markers`; only the "positive" list is used.
    ref_cell_type : str
        Column in `adata_ref.obs` containing cell-type labels.
    pos_threshold : float, optional
        Minimum fraction of cells within the target type where a gene must be present (>0).
    neg_threshold : float, optional
        Maximum fraction of cells in the complement (all other types) where the gene may be present.
    max_codetect : float, optional
        Maximum fraction of cells in which mutually exclusive gene pairs may be co-detected.
    cell_types : tuple[str, str], optional
        If provided, restrict computation to this pair of cell types.

    Returns
    -------
    list of tuple
        Pairs of genes (gene1, gene2) that are mutually exclusive across cell types.
    """
    # Extract positive marker genes for each cell type
    pos_by_ct = {ct: m.get("positive", []) for ct, m in markers.items()}

    # Flatten all genes across cell types, remove duplicates, and sort alphabetically
    all_genes = sorted({g for gs in pos_by_ct.values() for g in gs})

    # Keep only genes present in the AnnData object
    var_index = pd.Index(adata_ref.var_names)
    genes = [g for g in all_genes if g in var_index]
    if not genes:
        return []

    # Extract expression matrix for selected genes
    X = adata_ref[:, genes].X
    if sparse.issparse(X):
        X = X.tocsr()
        # Convert to binary presence/absence matrix (0/1)
        B = (X > 0).astype(np.uint8).tocsr()
    else:
        B = sparse.csr_matrix((np.asarray(X) > 0).astype(np.uint8))

    gene2col = {g: i for i, g in enumerate(genes)}  # map gene to column index
    labels = np.asarray(adata_ref.obs[ref_cell_type])
    cell_types_all = list(pos_by_ct.keys())  # all available cell types

    # === Restrict to user-specified cell types if provided ===
    if cell_types is not None:
        ct_subset = [ct for ct in cell_types if ct in cell_types_all]
        if len(ct_subset) != 2:
            raise ValueError(f"cell_types must contain exactly two valid types from: {cell_types_all}")
        cell_types_all = ct_subset

    # Dictionary to hold exclusive genes per cell type
    exclusive_genes = {ct: [] for ct in cell_types_all}
    all_exclusive = []

    n_cells = B.shape[0]  # total number of cells

    # === Step 1: Identify candidate exclusive genes per cell type ===
    for ct in cell_types_all:
        pos_genes = [g for g in pos_by_ct[ct] if g in gene2col]  # only genes in adata
        if not pos_genes:
            continue

        # Boolean masks for cells of this type vs all others
        mask_ct = labels == ct
        n_ct = int(mask_ct.sum())
        if n_ct == 0:
            continue
        mask_other = ~mask_ct
        n_other = int(mask_other.sum())

        # Subset binary matrix
        B_ct = B[mask_ct]
        B_other = B[mask_other]

        # Count number of cells where each gene is expressed
        ct_counts = np.asarray(B_ct.getnnz(axis=0)).ravel()
        other_counts = np.asarray(B_other.getnnz(axis=0)).ravel()

        # Fraction of cells expressing each gene
        frac_ct = ct_counts / max(n_ct, 1)
        frac_other = other_counts / max(n_other, 1)

        # Keep genes that are frequent in this type but rare in others
        idx = [gene2col[g] for g in pos_genes]
        keep = (frac_ct[idx] > pos_threshold) & (frac_other[idx] < neg_threshold)
        kept_genes = [g for g, k in zip(pos_genes, keep, strict=False) if k]

        exclusive_genes[ct] = kept_genes
        all_exclusive.extend(kept_genes)

    # === Step 2: Keep only genes that are exclusive to exactly one type ===
    freq = Counter(all_exclusive)
    unique_exclusive = {g for g, c in freq.items() if c == 1}
    filtered = {ct: [g for g in gs if g in unique_exclusive] for ct, gs in exclusive_genes.items()}

    # === Step 3: Form gene pairs ===
    if cell_types is not None:
        # Only generate pairs between the two user-specified types
        ct1, ct2 = cell_types
        pairs = [(g1, g2) for g1 in filtered.get(ct1, []) for g2 in filtered.get(ct2, [])]
    else:
        # Generate all cross-type pairs
        pairs = [
            (g1, g2) for ct1, ct2 in combinations(filtered.keys(), 2) for g1 in filtered[ct1] for g2 in filtered[ct2]
        ]

    # === Step 4: Filter pairs that are co-detected above threshold ===
    col_counts = np.asarray(B.getnnz(axis=0)).ravel()
    frac_overall = col_counts / max(n_cells, 1)

    def auto_pass(g1, g2):
        # If either gene is very rare overall, pair automatically passes
        return (frac_overall[gene2col[g1]] <= max_codetect) or (frac_overall[gene2col[g2]] <= max_codetect)

    trivial = [p for p in pairs if auto_pass(*p)]
    to_check = [p for p in pairs if not auto_pass(*p)]
    if not to_check:
        return trivial

    # Subset matrix to relevant columns for co-detection check
    B_csc = B.tocsc()
    cols_needed = np.array(sorted({gene2col[g] for p in to_check for g in p}), dtype=int)
    B_sub = B_csc[:, cols_needed]

    # Compute co-detection counts
    co_counts = (B_sub.T @ B_sub).tocsr()
    idx_map = {c: i for i, c in enumerate(cols_needed)}

    passed = []
    for g1, g2 in to_check:
        i = idx_map[gene2col[g1]]
        j = idx_map[gene2col[g2]]
        both = co_counts[i, j] / n_cells
        if both <= max_codetect:
            passed.append((g1, g2))

    # Return all passing mutually exclusive gene pairs
    result = trivial + passed

    return result


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
    labels_key: str = "cell_labels",
    labels_to_cell_id_key: str | None = "label_id",
    labels_data_key: str = None,
) -> bool:
    """
    Validates the integrity of a SpatialData object by checking the consistency of cell IDs
    across points, shapes, labels, and tables.

    This function ensures that:
    - All points have corresponding shapes, labels, and tables.
    - Cell IDs in points match those in shapes, labels, and tables.
    - If shapes or labels are present, they contain all cell IDs from the points.
    - If tables are present, they contain all cell IDs from the shapes.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object to validate.
    tables_key : str, optional
        Key for accessing tables in the SpatialData. Default is "table".
    tables_cell_id_key : str, optional
        Column name in the tables DataFrame (AnnData.obs) that contains cell IDs. Default is "cell_id".
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
    labels_key : str, optional
        Key for accessing segmentation labels in the SpatialData. Default is "cell_labels".
    labels_data_key : str, optional
        Key for accessing data within labels if they are stored as a DataTree. Default is None.
    labels_to_cell_id_key : str or None, optional
        Column in `sdata.tables[tables_key]` mapping segmentation label IDs
        (from `labels_key`) to cell IDs.

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
    # contains_labels = len(sdata.labels) > 0
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
    for coord_key in [points_x_key, points_y_key]:
        assert coord_key in points.columns, (
            f"Points DataFrame must contain coordinate column '{coord_key}'. "
            f"Available columns: {points.columns.tolist()}. "
            f"You can set this with the '{coord_key}' argument."
        )

    if points_z_key is not None:
        assert points_z_key in points.columns, (
            f"Points DataFrame must contain z coordinate column '{points_z_key}'. "
            f"Available columns: {points.columns.tolist()}. "
            f"You can set this with the '{coord_key}' argument."
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
        if labels_to_cell_id_key is not None:
            assert labels_to_cell_id_key in table.obs.columns, (
                f"Tables DataFrame must contain mapping to segmentation label IDs - '{tables_area_volume_key}'. "
                f"Available columns: {table.obs.columns.tolist()}. "
                f"You can set this with the 'labels_to_cell_id_key' argument (set to None if you do not have this)."
            )

    # get unique cell IDs from points
    transcript_ids = set(points[points_cell_id_key].unique())
    shapes_cell_ids = set()
    # labels_cell_ids = set()

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

        missing_in_polygons = {
            x
            for x in (transcript_ids - shapes_cell_ids - {points_background_id})
            if not _is_missing(x)  # also removing any NAs (no matter if from pandas, np, or None)
        }
        assert len(missing_in_polygons) == 0, (
            f"Missing {len(missing_in_polygons)} cell IDs from polygons: "
            f"{list(missing_in_polygons)[: min(5, len(missing_in_polygons))]}... "
            f"These cell IDs are present in the points, but not in the shapes. "
            f"If your missing cell ID is indicating an unassigned transcript, "
            f"you can set the points_background_id parameter."
        )

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

            tables_cell_ids = set(table.obs[tables_cell_id_key].values)

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

    # TODO: THIS NEEDS TO BE REACTIVATED AT SOME POINT
    # # if there are labels, ensure that there are no cell IDs in the points that are not in the labels
    # if contains_labels:
    #     # we can have multiple labels keys (e. g. when using multiple layers in proseg),
    # so we need to handle them here
    #     if isinstance(labels_key, str):
    #         assert labels_key in sdata.labels, (
    #             f"Labels DataFrame must contain key: {labels_key}. "
    #             f"Available keys: {list(sdata.labels.keys())}. "
    #             f"If you want to use a different key, set the labels_key parameter."
    #         )
    #         labels = sdata.labels[labels_key]

    #         # handling weird spatialdata structures
    #         if isinstance(labels, xr.DataTree):
    #             assert labels_data_key is not None, (
    #                 f"It looks like your labels are stored as a DataTree. "
    #                 f"Please provide a labels_data_key to access the labels data. "
    #                 f"Available keys are: {list(labels.keys())}."
    #             )
    #             assert labels_data_key.split("/")[0] in labels.keys(), (
    #                 f"Data key {labels_data_key} not found in the labels data. Available keys: {list(labels.keys())}"
    #             )

    #             labels = labels[labels_data_key]  # Get the dataset node

    #             assert isinstance(labels, xr.DataArray), (
    #                 f"The labels data should be a DataArray. Please provide a valid data key. "
    #                 f"Available keys are: {[labels_data_key + '/' + x for x in list(labels.keys())]}."
    #             )

    #             # label ID and cell ID are not the same
    #             labels_cell_ids = set(np.unique(labels)) - {0}  # Exclude background label (0)
    #     elif isinstance(labels_key, list):
    #         # if multiple labels keys are provided, we need to check each one
    #         labels_cell_ids = set()
    #         for key in labels_key:
    #             labels_tmp = sdata.labels[key]
    #             if isinstance(labels_tmp, xr.DataTree):
    #                 assert labels_data_key is not None, (
    #                     f"It looks like your labels are stored as a DataTree. "
    #                     f"Please provide a labels_data_key to access the labels data. "
    #                     f"Available keys are: {list(labels.keys())}."
    #                 )
    #                 labels_tmp = labels_tmp[labels_data_key]
    #                 assert isinstance(labels_tmp, xr.DataArray)
    #             # add nonzero unique labels
    #             labels_cell_ids.update(np.unique(labels_tmp).tolist())

    #         # Remove background label (e.g. 0)
    #         labels_cell_ids.discard(0)
    #     else:
    #         raise ValueError("labels_key must be a string or a list of strings")

    # # if there are both shapes and labels, ensure they are compatible
    # if contains_shapes and contains_labels:
    #     num_missing_in_shapes = len(labels_cell_ids) - len(shapes_cell_ids)
    #     num_missing_in_labels = len(shapes_cell_ids) - len(labels_cell_ids)
    #     if num_missing_in_labels > 0:
    #         warnings.warn(
    #             f"Missing {num_missing_in_labels} cell IDs in labels."
    #             f"There are {len(shapes_cell_ids)} cell IDs in shapes,
    # but only {len(labels_cell_ids)} are in labels. "
    #             f"This might lead to inconsistencies in the spatialdata object.",
    #             stacklevel=2,
    #         )
    #     if num_missing_in_shapes > 0:
    #         warnings.warn(
    #             f"Missing {num_missing_in_shapes} cell IDs in shapes: "
    #             f"There are {len(labels_cell_ids)} cell IDs in labels, but only
    # {len(shapes_cell_ids)} are in shapes. "
    #             f"This might lead to inconsistencies in the spatialdata object.",
    #             stacklevel=2,
    #         )

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
                f"You can set this with the 'nucleus_shapes_cell_id_key' argument "
                f"(set to None if you do not have this)."
            )

    return True
