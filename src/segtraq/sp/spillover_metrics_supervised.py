import warnings
from collections import Counter, defaultdict
from itertools import combinations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from scipy import sparse
from tqdm.auto import tqdm

from ..utils import _looks_like_counts


def _apply_overlap_filter(marker_dict: dict[str, list[str]], t, n_ct) -> dict[str, list[str]]:
    all_genes = [g for gl in marker_dict.values() for g in gl]
    if not all_genes:
        return {k: [] for k in marker_dict}
    counts = pd.Series(all_genes).value_counts()
    # drop genes appearing in >= t * n_types lists
    drop_genes = set(counts[counts >= (t * n_ct)].index)
    return {ct: [g for g in gl if g not in drop_genes] for ct, gl in marker_dict.items()}


def get_ref_markers(
    adata_ref: ad.AnnData,
    cell_type_column: str,
    q_pos: float = 0.95,
    q_neg: float = 0.10,
    t: float = 0.25,
) -> dict[str, dict[str, list[str]]]:
    """
    BIDCell/CellSPA-style marker discovery.

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
    cell_type_column : str
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
    ctypes = pd.Categorical(adata_ref.obs[cell_type_column])
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
    cell_type_column: str,
    pos_threshold: float = 0.20,
    neg_threshold: float = 0.05,
    max_codetect: float = 0.01,
    cell_types: tuple[str, str] | None = None,
) -> list[tuple[str, str]]:
    """
    Finds mutually exclusive markers (presence-based specificity) between cell types.

    Optionally restricts computation to a specified pair of cell types.

    Parameters
    ----------
    adata_ref : AnnData
        Reference single-cell dataset (cells × genes).
    markers : dict
        Marker dictionary as returned by `find_markers`; only the "positive" list is used.
    cell_type_column : str
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
    labels = np.asarray(adata_ref.obs[cell_type_column])
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
    return trivial + passed


def compute_MECR(sdata, gene_pairs: list[tuple[str, str]], table_key: str = "table") -> dict[tuple[str, str], float]:
    """
    Compute Mutually Exclusive Co-expression Rate (MECR) per gene pair.

    Parameters
    ----------
    sdata : SpatialData-like
        Container with `.tables[table_key]` as AnnData.
    gene_pairs : list of tuple
        Collection of (gene1, gene2) pairs.
    table_key : str
        Key of the AnnData table in `sdata.tables`.

    Returns
    -------
    dict
        Mapping {(gene1, gene2): MECR}, where MECR = P(both>0) / P(at least one>0).
    """
    mecr: dict[tuple[str, str], float] = {}
    expr_df = sdata.tables[table_key].to_df()

    for g1, g2 in gene_pairs:
        e1 = expr_df[g1] > 0
        e2 = expr_df[g2] > 0
        p_both = (e1 & e2).mean()
        p_any = (e1 | e2).mean()
        mecr[(g1, g2)] = (p_both / p_any) if p_any > 0 else 0.0

    return mecr


def calculate_contamination(
    sdata,
    markers,
    celltype_column: str,
    table_key: str = "table",
    radius: float = 15,
    n_neighs: int = 10,
    num_cells: int = 10_000,
    seed: int = 0,
    cell_centroid_x_key: str = "cell_centroid_x",
    cell_centroid_y_key: str = "cell_centroid_y",
    weight_edges: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute directional contamination (“leakage”) between cell types from spatial neighbors using
    ct-specific positive markers: for each ct cell and neighbor ct2, measure the fraction of ct-marker
    signal in ct2 neighbors relative to the ct neighborhood total, then average over interactions.

    Parameters
    ----------
    sdata : SpatialData-like
        Must contain `tables[table_key]` as AnnData with expression and coordinates.
    markers : dict
        {cell_type: {"positive": list[str], "negative": list[str] (optional)}}, using only "positive".
    celltype_column : str
        Column in the AnnData `.obs` with cell-type labels.
    table_key : str, optional
        Key of the AnnData table in `sdata.tables`.
    radius : float, optional
        Radius for spatial neighbor construction.
    n_neighs : int, optional
        Max number of neighbors per cell (used with `radius`).
    num_cells : int, optional
        Number of cells to sample (speed/precision trade-off).
    seed : int, optional
        RNG seed for reproducible sampling.
    cell_centroid_x_key : str, optional
        `.obs` key for x-coordinates (used to build `.obsm["spatial"]` if needed).
    cell_centroid_y_key : str, optional
        `.obs` key for y-coordinates (used to build `.obsm["spatial"]` if needed).
    weight_edges : bool, optional
        Weight neighbor contributions by graph edge weights if True.

    Returns
    -------
    out: pandas.DataFrame
        Rows = source types (ct), columns = target types (ct2); entry is the mean fraction of
        ct-specific marker counts found in ct2 neighbors relative to ct neighborhood totals (directional).
    records_df: pandas.DataFrame
        Pandas dataframe with per-cell cell_id (ct), cell_type (ct), neighbor_id (ct2), neigbhor_type (ct2) and ratio
        (ct-specific markers counts found in ct2 neighbors relative to ct neighborhood total counts).
    """

    adata = sdata.tables[table_key]

    adata.obsm["spatial"] = adata.obs[[cell_centroid_x_key, cell_centroid_y_key]].to_numpy()
    sq.gr.spatial_neighbors(adata, radius=radius, n_neighs=n_neighs, coord_type="generic")
    G = adata.obsp["spatial_connectivities"].tocsr()

    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(np.asarray(X))
    else:
        X = X.tocsr()

    # library sizes
    # libsize = np.asarray(X.sum(axis=1)).ravel()
    # mean_lib = float(libsize.mean())

    var_index = pd.Index(adata.var_names)
    pos_markers = {ct: [g for g in set(m.get("positive", [])) if g in var_index] for ct, m in markers.items()}
    gene_idx = {g: var_index.get_loc(g) for gs in pos_markers.values() for g in gs}

    ct_list = list(pos_markers)
    diff_cols = {}
    for ct in ct_list:
        set_ct = set(pos_markers[ct])
        for ct2 in ct_list:
            if ct == ct2:
                continue
            cols = [gene_idx[g] for g in set_ct.difference(pos_markers[ct2])]
            diff_cols[(ct, ct2)] = np.array(cols, dtype=int)

    rng = np.random.default_rng(seed)
    n = adata.n_obs
    idx_cells = rng.choice(n, size=min(num_cells, n), replace=False)

    types = np.asarray(adata.obs[celltype_column])
    C_sum = defaultdict(lambda: defaultdict(float))
    C_cnt = defaultdict(lambda: defaultdict(int))

    if weight_edges:
        # normalized weights can be derived from G; here we just use G as-is and add identity
        G_eff = G + sparse.eye(G.shape[0], format="csr")
    else:
        # unweighted: treat neighbors equally
        G_eff = None

    records = []

    for i in idx_cells:
        ct = types[i]

        start, end = G.indptr[i], G.indptr[i + 1]
        neigh = G.indices[start:end]
        if neigh.size == 0:
            continue

        for j in neigh:
            ct2 = types[j]
            if ct2 == ct:
                continue
            cols = diff_cols.get((ct, ct2))
            if cols is None or cols.size == 0:
                continue

            # numerator: neighbor j counts over S(ct, ct2) and scale by its total counts
            # TODO - check if this makes sense
            num = X[j, cols].sum()  # / (libsize[j] / mean_lib)

            # denominator: counts over S in (i ∪ N(i))
            if weight_edges:
                # weighted neighborhood sum: use (row i of G_eff) as weights
                w_idx = G_eff.indices[G_eff.indptr[i] : G_eff.indptr[i + 1]]
                w_val = G_eff.data[G_eff.indptr[i] : G_eff.indptr[i + 1]]
                # sum over rows w-weighted: (w^T @ X[:, cols]) -> use sparse vector-matrix product
                denom = (sparse.csr_matrix((w_val, ([0] * len(w_idx), w_idx)), shape=(1, n)) @ X[:, cols]).sum()
            else:
                # unweighted: sum rows {i} ∪ neigh
                denom = X[i, cols].sum() + X[neigh, :][:, cols].sum()

            denom = float(denom)
            if denom > 0.0:
                C_sum[ct][ct2] += float(num) / denom
                C_cnt[ct][ct2] += 1

                records.append(
                    {
                        "cell_id": int(i),
                        "cell_type": ct,
                        "neighbor_id": int(j),
                        "neighbor_type": ct2,
                        "ratio": float(num) / denom,
                    }
                )
                records_df = pd.DataFrame(records)

    cts = sorted(pos_markers)
    out = pd.DataFrame(0.0, index=cts, columns=cts)
    for ct in cts:
        for ct2 in cts:
            if ct == ct2:
                continue
            k = C_cnt[ct][ct2]
            out.loc[ct, ct2] = C_sum[ct][ct2] / k if k else 0.0

    out.index.name = "Source Cell Type"
    out.columns.name = "Target Cell Type"
    return C_cnt, out, records_df  # delete C_cnt later!!!


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


def calculate_marker_purity(
    sdata,
    celltype_column: str,
    markers: dict[str, dict[str, list[str]]],
    use_quantiles: bool = True,
    table_key: str = "table",
) -> pd.DataFrame:
    """
    Compute per-cell marker purity: for each cell's annotated type, evaluate Precision/Recall/F1
    using its positive and negative marker lists, then summarize into an overall `F1_purity`
    that rewards high positive-F1 and low negative-F1.

    Parameters
    ----------
    sdata : SpatialData-like
        Must contain `tables[table_key]` as an AnnData with expression and `.obs` metadata.
    celltype_column : str
        Column in the AnnData `.obs` with cell-type labels.
    markers : dict
        {cell_type: {"positive": list[str], "negative": list[str]}}; both lists are required.
    use_quantiles : bool, optional
        If True, define predictions by the top-|markers| fraction per cell (rank-based);
        if False, use direct expression-based criteria (e.g., >0).
    table_key : str, optional
        Key of the AnnData table in `sdata.tables`.

    Returns
    -------
    pandas.DataFrame
        Columns: ['positive_precision','positive_recall','positive_F1',
                  'negative_precision','negative_recall','negative_F1',
                  'F1_purity','cell_type'] indexed by cell.
    """

    adata = sdata.tables[table_key]

    # dense view for quantiles; adjust if you need to stay sparse
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    genes = np.asarray(adata.var_names)
    cell_types = np.asarray(adata.obs[celltype_column])
    n_cells, n_genes = X.shape

    def _idx(lst: list[str]) -> np.ndarray:
        if not lst:
            return np.empty(0, dtype=int)
        return np.where(np.isin(genes, np.asarray(lst)))[0]

    pos_idx_map = {ct: _idx(m.get("positive", [])) for ct, m in markers.items()}
    neg_idx_map = {ct: _idx(m.get("negative", [])) for ct, m in markers.items()}

    rows = []
    for i in range(n_cells):
        ct = cell_types[i]
        expr = X[i, :]

        # positive pass (upper quantile)
        p_prec, p_rec, p_f1 = _score_one_list(expr, pos_idx_map.get(ct, np.empty(0, dtype=int)), n_genes, use_quantiles)

        # negative pass (upper quantile)
        n_prec, n_rec, n_f1 = _score_one_list(expr, neg_idx_map.get(ct, np.empty(0, dtype=int)), n_genes, use_quantiles)

        # fused purity
        denom = (1.0 - n_f1) + p_f1
        f1_purity = (2.0 * (1.0 - n_f1) * p_f1 / denom) if denom > 0 else 0.0

        rows.append(
            {
                "positive_precision": p_prec,
                "positive_recall": p_rec,
                "positive_F1": p_f1,
                "negative_precision": n_prec,
                "negative_recall": n_rec,
                "negative_F1": n_f1,
                "F1_purity": f1_purity,
            }
        )

    return pd.DataFrame(rows, index=adata.obs["cell_id"])


def calculate_diff_abundance(
    sdata,
    celltype_column: str,
    markers: dict[str, dict[str, list[str]]],
    table_key: str = "table",
    lfc_thresh: float = 1.0,  # noqa
    pval_thresh: float = 0.05,  # noqa
    min_n_cells: int = 20,
    min_n_transcripts: int = 20,
    seed: int = 0,
    cell_centroid_x_key: str = "cell_centroid_x",
    cell_centroid_y_key: str = "cell_centroid_y",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate differential transcript abundance between bordering and non-bordering cells
    for every ordered pair of cell types using a spatial graph (Delaunay triangulation).
    This function builds a spatial neighbor graph from the provided AnnData table, classifies
    cells of each source cell type (ct1) into two groups depending on whether they border
    cells of a target cell type (ct2), and performs a differential abundance test (Scanpy's
    rank_genes_groups with Wilcoxon test) between the "bordering" and "non_bordering" groups.
    Results are filtered by transcript counts and by provided marker lists so that only genes
    likely to originate from spillover (positive markers of the source cell type ct2 and
    not negative markers of the receiver ct1) are retained. A summary table of significant
    genes per (ct1, ct2) pair is returned.

    Parameters
    ----------
    sdata : object
        Object containing AnnData tables, expected to expose a mapping-like attribute
        `tables` such that `sdata.tables[table_key]` is an AnnData instance. The function
        creates a local copy of that AnnData and operates on it.
    celltype_column : str
        Column name in adata.obs that contains cell type labels.
    markers : dict[str, dict[str, list[str]]]
        Marker specification mapping cell type -> {"positive": [...], "negative": [...]}
        - markers[ct]['positive'] should list genes expected to be present in source cells (ct2).
        - markers[ct]['negative'] should list genes expected to be absent in receiver cells (ct1).
    table_key : str, optional
        Key to select the AnnData table from sdata.tables (default: "table").
    min_n_cells : int, optional
        Minimum number of cells required in each group (bordering or non_bordering)
        for a (ct1, ct2) pair to be tested (default: 20).
    min_n_transcripts : int, optional
        Minimum total transcript counts (sum across both groups) required for a gene
        to be kept in the results (default: 20).
    seed : int, optional
        Random seed for reproducible subsampling (default: 0).
    cell_centroid_x_key, cell_centroid_y_key : str, optional
        Column names in adata.obs that contain the X and Y cell centroids used to
        construct the spatial graph (default: "cell_centroid_x", "cell_centroid_y").

    Returns
    -------
    de_results : pandas.DataFrame
        Concatenated differential abundance results for all tested (ct1, ct2) pairs.
        Columns include:
        - gene: gene name (string)
        - log2FC: reported log fold-change from rank_genes_groups
        - pval: p-value from the differential test
        - ct1: receiver cell type (string)
        - ct2: source cell type (string)
        - group1_size: number of ct1 cells bordering ct2 (int)
        - group2_size: number of ct1 cells not bordering ct2 (int)
        - transcript_counts_group1: total counts of the gene across group1 (int)
        - transcript_counts_group2: total counts of the gene across group2 (int)
        - transcript_counts_in_both_groups: sum of the two previous columns (int)
    summary : pandas.DataFrame
        A matrix (DataFrame) where rows are receiver cell types (ct1) and columns are
        source cell types (ct2). Each cell contains the number of genes passing the
        significance criteria for that ordered pair. Note: significance filtering uses
        the thresholds lfc_thresh and pval_thresh from the calling/global scope
        (see Notes).
    Raises
    ------
    ValueError
        If no differential expression results are produced (e.g., because no
        (ct1, ct2) pairs passed the min_n_cells / marker / transcript filters).
    """
    adata = sdata.tables[table_key].copy()
    adata.obsm["spatial"] = adata.obs[[cell_centroid_x_key, cell_centroid_y_key]].to_numpy()

    # Replace NA cell types
    col = adata.obs[celltype_column]
    if pd.api.types.is_categorical_dtype(col):
        if "Unknown" not in col.cat.categories:
            col = col.cat.add_categories(["Unknown"])
        adata.obs[celltype_column] = col.fillna("Unknown")
    else:
        adata.obs[celltype_column] = col.fillna("Unknown")

    # 1. Build spatial graph (Delaunay triangulation)
    sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")
    G = adata.obsp["spatial_connectivities"].tocsr()

    types = np.asarray(adata.obs[celltype_column])
    cell_types = np.unique(types)

    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()

    de_records = []

    # 2. Iterate over cell-type pairs
    for ct1 in tqdm(cell_types):
        idx_ct1 = np.where(types == ct1)[0]
        for ct2 in cell_types:
            if ct1 == ct2:
                continue

            # Find ct1 cells with ct2 neighbors
            neigh_counts = np.array([np.any(types[G.indices[G.indptr[i] : G.indptr[i + 1]]] == ct2) for i in idx_ct1])
            group1 = idx_ct1[neigh_counts]  # bordering
            group2 = idx_ct1[~neigh_counts]  # non-bordering
            group1_size, group2_size = len(group1), len(group2)

            if group1_size < min_n_cells or group2_size < min_n_cells:
                continue

            # Create condition labels
            adata.obs["_temp_condition"] = "not_used"
            adata.obs.iloc[group1, adata.obs.columns.get_loc("_temp_condition")] = "bordering"
            adata.obs.iloc[group2, adata.obs.columns.get_loc("_temp_condition")] = "non_bordering"

            # Differential test
            sc.tl.rank_genes_groups(
                adata,
                use_raw=False,
                groupby="_temp_condition",
                groups=["bordering"],
                reference="non_bordering",
                method="wilcoxon",
                pts=True,
            )

            res_dict = adata.uns["rank_genes_groups"]

            # ensure everything is in the right order
            genes = res_dict["names"]["bordering"]
            log2fc = res_dict["logfoldchanges"]["bordering"]
            pval = res_dict["pvals"]["bordering"]

            # transcript counts per group (while ensuring this is in the correct order)
            gene_idx = [adata.var_names.get_loc(g) for g in genes]
            transcript_counts_group1 = np.array(adata.raw.X[group1, :][:, gene_idx].sum(axis=0)).ravel()
            transcript_counts_group2 = np.array(adata.raw.X[group2, :][:, gene_idx].sum(axis=0)).ravel()

            res = pd.DataFrame(
                {
                    "gene": genes,
                    "log2FC": log2fc,
                    "pval": pval,
                    "ct1": ct1,
                    "ct2": ct2,
                    "group1_size": group1_size,
                    "group2_size": group2_size,
                    "transcript_counts_group1": transcript_counts_group1,
                    "transcript_counts_group2": transcript_counts_group2,
                    "transcript_counts_in_both_groups": transcript_counts_group1 + transcript_counts_group2,
                }
            )

            # removing rows with transcript counts lower than a certain minimum
            res = res[res["transcript_counts_in_both_groups"] >= min_n_transcripts]

            # only keeping genes that are positive in the source (ct2) and negative in the receiver (ct1)
            try:
                ct1_markers = markers[ct1]["negative"]
            except KeyError:
                if ct1 != "Unknown":
                    print(f"Could not find markers for cell type {ct1}")
                ct1_markers = []
            res = res[~res["gene"].isin(ct1_markers)]

            try:
                ct2_markers = markers[ct2]["positive"]
            except KeyError:
                if ct2 != "Unknown":
                    print(f"Could not find markers for cell type {ct2}")
                ct2_markers = []
            res = res[res["gene"].isin(ct2_markers)]

            de_records.append(res)

    # Combine results
    if not de_records:
        raise ValueError("No DE results produced — check thresholds or data sparsity.")
    de_results = pd.concat(de_records, ignore_index=True)

    # 4. Summarize significant DE genes
    # we are only interested in positive log2FC here, since we want to find genes that originate from spillover events
    sig = de_results.query("log2FC >= @lfc_thresh and pval <= @pval_thresh")
    summary = sig.groupby(["ct1", "ct2"]).size().unstack(fill_value=0)

    return de_results, summary
