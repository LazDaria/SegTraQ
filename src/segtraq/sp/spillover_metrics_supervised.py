import pandas as pd
import numpy as np
import anndata as ad
from typing import Dict, List, Tuple
import scanpy as sc
from itertools import combinations
import squidpy as sq
from collections import Counter
import warnings
from ..utils import _looks_like_counts
from scipy import sparse

def _apply_overlap_filter(marker_dict: Dict[str, List[str]], t, n_ct) -> Dict[str, List[str]]:
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
) -> Dict[str, Dict[str, List[str]]]:
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
        mask = (ctypes == ct)
        if mask.sum() == 0:
            means[ct] = np.zeros(adata_ref.n_vars, dtype=float)
        else:
            means[ct] = X[mask].mean(axis=0)
    ref_exprs = pd.DataFrame(means, index=genes)

    # differential score w = mean_in_type - mean_in_others
    pos_lists: Dict[str, List[str]] = {}
    neg_lists: Dict[str, List[str]] = {}
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
    adata_ref: ad.AnnData,
    markers: Dict[str, Dict[str, List[str]]],
    cell_type_column: str,
    pos_threshold: float = 0.20,
    neg_threshold: float = 0.05,
) -> List[Tuple[str, str]]:
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
    Returns
    -------
    list of tuple
        Pairs of genes (gene1, gene2) that are mutually exclusive across cell types.
    """

    pos_by_ct = {ct: m.get("positive", []) for ct, m in markers.items()}
    all_genes = sorted({g for gs in pos_by_ct.values() for g in gs})

    var_index = pd.Index(adata_ref.var_names)
    genes = [g for g in all_genes if g in var_index]


    X = adata_ref[:, genes].X
    if sparse.issparse(X):
        X = X.tocsr()
        B = (X > 0).tocsr()
    else:
        B = (np.asarray(X) > 0)

    gene2col = {g: i for i, g in enumerate(genes)}

    labels = np.asarray(adata_ref.obs[cell_type_column])
    cell_types = list(pos_by_ct.keys())

    exclusive_genes = {ct: [] for ct in cell_types}
    all_exclusive = []

    for ct in cell_types:
        pos_genes = [g for g in pos_by_ct[ct] if g in gene2col]
        if not pos_genes:
            continue

        mask_ct = (labels == ct)
        n_ct = int(mask_ct.sum())
        if n_ct == 0:
            continue
        mask_other = ~mask_ct
        n_other = int(mask_other.sum())

        if sparse.issparse(B):
            ct_counts = np.asarray(B[mask_ct].getnnz(axis=0)).ravel()
            other_counts = np.asarray(B[mask_other].getnnz(axis=0)).ravel()
        else:
            ct_counts = B[mask_ct].sum(axis=0).ravel()
            other_counts = B[mask_other].sum(axis=0).ravel()

        frac_ct = ct_counts / max(n_ct, 1)
        frac_other = other_counts / max(n_other, 1)

        idx = [gene2col[g] for g in pos_genes]
        keep = (frac_ct[idx] > pos_threshold) & (frac_other[idx] < neg_threshold)
        kept_genes = [g for g, k in zip(pos_genes, keep) if k]

        exclusive_genes[ct] = kept_genes
        all_exclusive.extend(kept_genes)

    freq = Counter(all_exclusive)
    unique_exclusive = {g for g, c in freq.items() if c == 1}
    filtered = {ct: [g for g in gs if g in unique_exclusive] for ct, gs in exclusive_genes.items()}

    pairs = [(g1, g2)
             for ct1, ct2 in combinations(filtered.keys(), 2)
             for g1 in filtered[ct1] for g2 in filtered[ct2]]
    return pairs

def compute_MECR(
    sdata,
    gene_pairs: List[Tuple[str, str]],
    table_key: str = "table"
) -> Dict[Tuple[str, str], float]:
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
    mecr: Dict[Tuple[str, str], float] = {}
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
    markers: Dict[str, Dict[str, List[str]]],
    radius: float = 15,
    n_neighs: int = 10,
    celltype_column: str = "celltype_major",
    num_cells: int = 10000,
    table_key: str = "table",
    centroid_x_key: str = "x_centroid",
    centroid_y_key: str = "y_centroid"
) -> pd.DataFrame:
    """
    Estimate normalized cross-type contamination using positive markers and spatial neighborhoods.

    Parameters
    ----------
    sdata : SpatialData-like
        Container with `.tables[table_key]` as AnnData.
    markers : dict
        Marker dictionary with per-type 'positive' and 'negative' lists.
    radius : float, optional
        Neighborhood radius for spatial graph (default: 15).
    n_neighs : int, optional
        Max neighbors to retain (default: 10).
    celltype_column : str, optional
        Column in `adata.obs` with cell type labels (default: 'celltype_major').
    num_cells : int, optional
        Number of randomly sampled cells used for estimation (default: 10000).
    table_key : str, optional
        Key of the AnnData table in `sdata.tables`.
    centroid_x_key: str, optional
        Column in .tables[table_key]` with spatial x centroids.
    centroid_y_key: str, optional
        Column in .tables[table_key]` with spatial y centroids.
    
    Returns
    -------
    pandas.DataFrame
        Matrix of normalized contamination from source → target types.

    Raises
    ------
    ValueError
        If `celltype_column` is not present in `adata.obs`.
    """
    adata = sdata.tables[table_key]

    if celltype_column not in adata.obs:
        raise ValueError("Column celltype_column must be present in adata.obs.")

    # Positive marker lookup per cell type
    pos_markers = {ct: markers[ct]["positive"] for ct in markers}

    # Build spatial graph from centroids
    adata.obsm["spatial"] = adata.obs[[centroid_x_key, centroid_y_key]].to_numpy()
    sq.gr.spatial_neighbors(adata, radius=radius, n_neighs=n_neighs, coord_type="generic")
    neighbors = adata.obsp["spatial_connectivities"].tolil()

    # Raw counts matrix and metadata
    raw = adata[:, adata.var_names].layers["raw"].toarray() #####TODO - handle check - whether log-normalized and store counts in raw
    cell_types = adata.obs[celltype_column].values

    # Sampling set
    sel = np.random.choice(adata.n_obs, size=min(num_cells, adata.n_obs), replace=False)

    contamination: Dict[str, Dict[str, float]] = {ct: {ct2: 0.0 for ct2 in pos_markers} for ct in pos_markers}
    negighborings: Dict[str, Dict[str, int]] = {ct: {ct2: 0 for ct2 in pos_markers} for ct in pos_markers}

    # Iterate over sampled cells
    for idx in sel:
        src_type = cell_types[idx]
        own = set(pos_markers[src_type])

        # Sum marker counts within the local neighborhood (including self)
        for marker in own:
            if marker in adata.var_names:
                total_in_nbhd = raw[idx, adata.var_names.get_loc(marker)]
                for nb in neighbors.rows[idx]:
                    total_in_nbhd += raw[nb, adata.var_names.get_loc(marker)]

                # Attribute neighbor-specific fractions to cross-type contamination
                for nb in neighbors.rows[idx]:
                    nb_type = cell_types[nb]
                    if nb_type == src_type:
                        continue

                    nb_markers = set(pos_markers.get(nb_type, []))
                    contam_markers = own - nb_markers  # avoid markers shared with neighbor's own type

                    for m in contam_markers:
                        if m in adata.var_names:
                            m_counts_nb = raw[nb, adata.var_names.get_loc(m)]
                            if total_in_nbhd > 0:
                                contamination[src_type][nb_type] += m_counts_nb / total_in_nbhd
                                negighborings[src_type][nb_type] += 1

    # Normalize by neighbor-count accumulator (with +1 as in original)
    contam_df = pd.DataFrame(contamination).T
    neg_df = pd.DataFrame(negighborings).T
    contam_df.index.name = "Source Cell Type"
    contam_df.columns.name = "Target Cell Type"

    return contam_df / (neg_df + 1)

def calculate_sensitivity(
    sdata,
    purified_markers: Dict[str, List[str]],
    max_cells_per_type: int = 1000,
    table_key: str = "table",
    celltype_column : str = "celltype_major"
) -> Dict[str, List[float]]:
    """
    Compute per-cell sensitivity of purified marker sets for each cell type.

    Parameters
    ----------
    sdata : SpatialData-like
        Container with `.tables[table_key]` as AnnData.
    purified_markers : dict
        Mapping {cell_type: {'positive': List[str], ...}} of purified markers.
    max_cells_per_type : int, optional
        Cap on cells per type when computing sensitivities (default: 1000).
    table_key : str, optional
        Key of the AnnData table in `sdata.tables`.
    celltype_column : str, optional
        Column in `adata.obs` with cell type labels (default: 'celltype_major').

    Returns
    -------
    dict
        {cell_type: List[float]} where each value is the fraction of markers expressed in a cell.
    """
    adata = sdata.tables[table_key]

    results: Dict[str, List[float]] = {ct: [] for ct in purified_markers}

    for ct, mk in purified_markers.items():
        pos = mk["positive"]
        subset = adata[adata.obs[celltype_column] == ct]

        # Optional downsampling
        if subset.n_obs > max_cells_per_type:
            idx = np.random.choice(subset.n_obs, max_cells_per_type, replace=False)
            subset = subset[idx]

        # Fraction of positive markers expressed per cell
        if len(pos) == 0:
            results[ct].extend([0.0] * subset.n_obs)
            continue

        pos_idx = subset.var_names.get_indexer(pos)
        for row in subset.X.toarray():
            n_expr = np.asarray((row[pos_idx] > 0).sum())
            results[ct].append(n_expr / len(pos))

    return results

def _score_one_list(expr: np.ndarray, marker_idx: np.ndarray, n_genes: int, use_quantiles: bool) -> tuple:
    """Precision, recall, F1 for one list using upper-quantile rule (CellSPA)."""
    if marker_idx.size == 0:
        return 0.0, 0.0, 0.0

    actual = np.zeros(n_genes, dtype=bool)
    actual[marker_idx] = True
    frac = actual.mean()  # |markers| / G

    if use_quantiles and frac > 0:
        thr = np.quantile(expr, 1.0 - frac)  # upper quantile
        predicted = expr > thr
    else:
        # fallback (CellSPA always uses quantiles; this branch keeps API flexible)
        predicted = expr > 0

    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, F1

def calculate_marker_purity(
    sdata,
    celltype_column: str,
    markers: Dict[str, Dict[str, List[str]]],
    use_quantiles: bool = True,
    table_key: str = "table",
) -> pd.DataFrame:
    """
    CellSPA-style marker purity for positive & negative lists from a single dict.

    markers[ct] must be like: {"positive": [...], "negative": [...]}

    For each cell with type `ct`:
      - Positive pass: score Precision/Recall/F1 using markers[ct]["positive"]
        Predicted = top-|pos|/G fraction of genes (upper quantile).
      - Negative pass: score Precision/Recall/F1 using markers[ct]["negative"]
        Predicted = top-|neg|/G fraction of genes (upper quantile)  <-- matches CellSPA’s call with a “negative” list.
      - Purity summary (optional): F1_purity = 2 * ((1 - F1_neg) * F1_pos) / ((1 - F1_neg) + F1_pos)

    Returns a DataFrame indexed by cells with columns:
      ['positive_precision','positive_recall','positive_F1',
       'negative_precision','negative_recall','negative_F1',
       'F1_purity','cell_type']
    """
    adata = sdata.tables[table_key]

    # dense view for quantiles; adjust if you need to stay sparse
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    genes = np.asarray(adata.var_names)
    cell_types = np.asarray(adata.obs[celltype_column])
    n_cells, n_genes = X.shape

    def _idx(lst: List[str]) -> np.ndarray:
        if not lst:
            return np.empty(0, dtype=int)
        return np.where(np.isin(genes, np.asarray(lst)))[0]

    pos_idx_map = {ct: _idx(m.get("positive", [])) for ct, m in markers.items()}
    neg_idx_map = {ct: _idx(m.get("negative", [])) for ct, m in markers.items()}

    rows = []
    for i in range(n_cells):
        ct   = cell_types[i]
        expr = X[i, :]

        # positive pass (upper quantile)
        p_prec, p_rec, p_f1 = _score_one_list(expr, pos_idx_map.get(ct, np.empty(0, dtype=int)), n_genes, use_quantiles)

        # negative pass (upper quantile as well — same as CellSPA called with a negative list)
        n_prec, n_rec, n_f1 = _score_one_list(expr, neg_idx_map.get(ct, np.empty(0, dtype=int)), n_genes, use_quantiles)

        # fused purity (optional summary)
        denom = (1.0 - n_f1) + p_f1
        f1_purity = (2.0 * (1.0 - n_f1) * p_f1 / denom) if denom > 0 else 0.0

        rows.append({
            "positive_precision": p_prec,
            "positive_recall": p_rec,
            "positive_F1": p_f1,
            "negative_precision": n_prec,
            "negative_recall": n_rec,
            "negative_F1": n_f1,
            "F1_purity": f1_purity
        })

    return pd.DataFrame(rows, index=adata.obs["cell_id"])
