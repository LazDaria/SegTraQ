import warnings
from collections import Counter, defaultdict
from itertools import combinations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from scipy import sparse

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
) -> list[tuple[str, str]]:
    """
    Finds mutually exclusive markers (presence-based specificity)

    For each cell type c, scan its positive markers and keep genes that are present
    in > pos_threshold of cells of c and < neg_threshold of cells in all other types.
    From these candidates, retain only genes that satisfy the rule for a single cell type;
    finally, return all cross-type pairs formed by these type-unique genes.

    Parameters
    ----------
    adata_ref : AnnData
        Reference single-cell dataset (cells × genes).
    markers : dict
        Marker dictionary as returned by `find_markers`; only the "positive" list is used.
    cell_type_column : str
        Column in `adata_ref.obs` containing cell-type labels.
    pos_threshold : float, optional (default: 0.20)
        Minimum fraction of cells within the target type where a gene must be present (>0).
    neg_threshold : float, optional (default: 0.05)
        Maximum fraction of cells in the complement (all other types) where the gene may be present.
    max_codetect: float, optional (default: 0.01)
        Maximum fraction of cells in which mutually exclusive gene pairs may be co-detected.
    Returns
    -------
    list of tuple
        Pairs of genes (gene1, gene2) that are mutually exclusive across cell types.
    """
    pos_by_ct = {ct: m.get("positive", []) for ct, m in markers.items()}
    all_genes = sorted({g for gs in pos_by_ct.values() for g in gs})
    var_index = pd.Index(adata_ref.var_names)
    genes = [g for g in all_genes if g in var_index]
    if not genes:
        return []

    X = adata_ref[:, genes].X
    if sparse.issparse(X):
        X = X.tocsr()
        B = (X > 0).astype(np.uint8).tocsr()
    else:
        B = sparse.csr_matrix((np.asarray(X) > 0).astype(np.uint8))

    gene2col = {g: i for i, g in enumerate(genes)}
    labels = np.asarray(adata_ref.obs[cell_type_column])
    cell_types = list(pos_by_ct.keys())

    exclusive_genes = {ct: [] for ct in cell_types}
    all_exclusive = []

    n_cells = B.shape[0]
    for ct in cell_types:
        pos_genes = [g for g in pos_by_ct[ct] if g in gene2col]
        if not pos_genes:
            continue

        mask_ct = labels == ct
        n_ct = int(mask_ct.sum())
        if n_ct == 0:
            continue
        mask_other = ~mask_ct
        n_other = int(mask_other.sum())

        B_ct = B[mask_ct]
        B_other = B[mask_other]

        ct_counts = np.asarray(B_ct.getnnz(axis=0)).ravel()
        other_counts = np.asarray(B_other.getnnz(axis=0)).ravel()

        frac_ct = ct_counts / max(n_ct, 1)
        frac_other = other_counts / max(n_other, 1)

        idx = [gene2col[g] for g in pos_genes]
        keep = (frac_ct[idx] > pos_threshold) & (frac_other[idx] < neg_threshold)
        kept_genes = [g for g, k in zip(pos_genes, keep, strict=False) if k]

        exclusive_genes[ct] = kept_genes
        all_exclusive.extend(kept_genes)

    # keep genes that are exclusive to exactly one type
    freq = Counter(all_exclusive)
    unique_exclusive = {g for g, c in freq.items() if c == 1}
    filtered = {ct: [g for g in gs if g in unique_exclusive] for ct, gs in exclusive_genes.items()}

    pairs = [(g1, g2) for ct1, ct2 in combinations(filtered.keys(), 2) for g1 in filtered[ct1] for g2 in filtered[ct2]]

    # filter out genes that are co-detected in >=max_codetect
    col_counts = np.asarray(B.getnnz(axis=0)).ravel()
    frac_overall = col_counts / max(n_cells, 1)

    # pre-filter (if either gene is present in <= max_codetect of cells, pair cannot exceed the threshold)
    def auto_pass(g1, g2):
        return (frac_overall[gene2col[g1]] <= max_codetect) or (frac_overall[gene2col[g2]] <= max_codetect)

    trivial = [p for p in pairs if auto_pass(*p)]
    to_check = [p for p in pairs if not auto_pass(*p)]
    if not to_check:
        return trivial

    B_csc = B.tocsc()

    cols_needed = np.array(sorted({gene2col[g] for p in to_check for g in p}), dtype=int)
    B_sub = B_csc[:, cols_needed]

    co_counts = (B_sub.T @ B_sub).tocsr()

    idx_map = {c: i for i, c in enumerate(cols_needed)}
    passed = []
    for g1, g2 in to_check:
        i = idx_map[gene2col[g1]]
        j = idx_map[gene2col[g2]]
        both = co_counts[i, j] / n_cells
        if both <= max_codetect:
            passed.append((g1, g2))

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
