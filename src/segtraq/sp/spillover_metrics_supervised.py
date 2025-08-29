import pandas as pd
import numpy as np
import anndata as ad
from scipy.spatial import ConvexHull
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import scanpy as sc
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import dask
import squidpy as sq

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import anndata as ad

def find_markers_cellspa(
    adata_ref: ad.AnnData,
    cell_type_column: str,
    q: float = 0.90,
    t: float = 0.25,
) -> Dict[str, Dict[str, List[str]]]:
    """
    BIDCell/CellSPA-style marker discovery with Segger-style output.

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
    q : float, optional (default: 0.90)
        Upper quantile for positives; lower (1 - q) is used for negatives.
    t : float, optional (default: 0.25)
        Overlap filter: drop genes that appear in >= t * n_types marker lists.

    Returns
    -------
    dict
        {cell_type: {"positive": [genes], "negative": [genes]}}
    """

    X = adata_ref.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    genes = np.asarray(adata_ref.var_names)
    ctypes = pd.Categorical(adata_ref.obs[cell_type_column])
    types = list(ctypes.categories)
    n_types = len(types)
    if n_types < 2:
        raise ValueError("Need at least two cell types to compute differential markers.")

    # --- compute per-type mean expression (genes x types)
    # result: DataFrame with index=genes, columns=types
    means = {}
    for ct in types:
        mask = (ctypes == ct)
        if mask.sum() == 0:
            means[ct] = np.zeros(adata_ref.n_vars, dtype=float)
        else:
            means[ct] = X[mask].mean(axis=0)
    ref_exprs = pd.DataFrame(means, index=genes)

    # --- differential score w = mean_in_type - mean_in_others
    pos_lists: Dict[str, List[str]] = {}
    neg_lists: Dict[str, List[str]] = {}
    type_cols = ref_exprs.columns.to_list()

    for ct in type_cols:
        in_ct = ref_exprs[ct].to_numpy()
        others = ref_exprs.drop(columns=[ct]).mean(axis=1).to_numpy()
        w = in_ct - others

        # quantile cutoffs
        q_hi = np.quantile(w, q)
        q_lo = np.quantile(w, 1.0 - q)

        # positives = top-q
        pos_genes = ref_exprs.index[w > q_hi].tolist()
        # negatives = bottom-q
        neg_genes = ref_exprs.index[w < q_lo].tolist()

        pos_lists[ct] = pos_genes
        neg_lists[ct] = neg_genes

    # --- overlap filter (remove ubiquitous markers)
    def _apply_overlap_filter(marker_dict: Dict[str, List[str]], t) -> Dict[str, List[str]]:
        all_genes = [g for gl in marker_dict.values() for g in gl]
        if not all_genes:
            return {k: [] for k in marker_dict}
        counts = pd.Series(all_genes).value_counts()
        # drop genes appearing in >= t * n_types lists
        drop_genes = set(counts[counts >= (t * n_types)].index)
        return {ct: [g for g in gl if g not in drop_genes] for ct, gl in marker_dict.items()}

    pos_lists = _apply_overlap_filter(pos_lists, t=t)
    neg_lists = _apply_overlap_filter(neg_lists, t=1)

    # --- assemble Segger-style output
    markers = {ct: {"positive": pos_lists.get(ct, []), "negative": neg_lists.get(ct, [])} for ct in types}
    return markers


def find_markers(
    adata_ref: ad.AnnData,
    cell_type_column: str,
    pos_percentile: float = 5,
    neg_percentile: float = 10,
    percentage: float = 50,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Derive positive/negative marker sets per cell type using percentile cutoffs and an expression fraction filter.

    Parameters
    ----------
    adata_ref : AnnData
        Reference dataset.
    cell_type_column : str
        Column in `adata.obs` indicating cell type labels.
    pos_percentile : float, optional
        Upper percentile defining highly expressed genes (default: 5).
    neg_percentile : float, optional
        Lower percentile defining lowly expressed genes (default: 10).
    percentage : float, optional
        Minimum % of cells within a type that must express a positive marker (default: 50).

    Returns
    -------
    dict
        Mapping {cell_type: {'positive': List[str], 'negative': List[str]}}.
    """
    markers: Dict[str, Dict[str, List[str]]] = {}

    sc.tl.rank_genes_groups(adata_ref, groupby=cell_type_column)

    genes = adata_ref.var_names
    for cell_type in adata_ref.obs[cell_type_column].unique():
        # Subset to a single cell type
        subset = adata_ref[adata_ref.obs[cell_type_column] == cell_type]

        # Mean expression per gene and percentile thresholds
        mean_expr = np.asarray(subset.X.mean(axis=0)).flatten()
        hi_cut = np.percentile(mean_expr, 100 - pos_percentile)
        lo_cut = np.percentile(mean_expr, neg_percentile)

        # Indices for positive/negative sets by thresholds
        pos_idx = np.where(mean_expr >= hi_cut)[0]
        neg_idx = np.where(mean_expr <= lo_cut)[0]

        # Enforce within-type expression fraction for positives
        expr_frac = np.asarray((subset.X[:, pos_idx] > 0).mean(axis=0)).flatten()
        valid_pos_idx = pos_idx[expr_frac >= (percentage / 100)]

        positive_markers = list(genes[valid_pos_idx])
        negative_markers = list(genes[neg_idx])

        markers[cell_type] = {"positive": positive_markers, "negative": negative_markers}

    return markers


def find_mutually_exclusive_genes(
    adata_ref: ad.AnnData,
    markers: Dict[str, Dict[str, List[str]]],
    cell_type_column: str,
) -> List[Tuple[str, str]]:
    """
    Extract mutually exclusive gene pairs based on expression specificity criteria.

    Parameters
    ----------
    adata_ref : AnnData
        Reference dataset.
    markers : dict
        Marker dictionary as returned by `find_markers`.
    cell_type_column : str
        Column in `adata.obs` indicating cell type labels.

    Returns
    -------
    list of tuple
        Pairs of genes (gene1, gene2) that are mutually exclusive across cell types.
    """
    exclusive_genes: Dict[str, List[str]] = {}
    all_exclusive: List[str] = []

    for cell_type, marker_sets in markers.items():
        positive = marker_sets["positive"]
        exclusive_genes[cell_type] = []

        for gene in positive:
            gene_expr = adata_ref[:, gene].X
            mask_ct = (adata_ref.obs[cell_type_column] == cell_type).to_numpy(dtype=bool)
            mask_other = ~mask_ct

            # Specificity rule: present in >20% of target type, <5% of others
            if (gene_expr[mask_ct] > 0).mean() > 0.2 and (gene_expr[mask_other] > 0).mean() < 0.05:
                exclusive_genes[cell_type].append(gene)
                all_exclusive.append(gene)

    # Keep only genes that appear in the exclusive lists
    unique_exclusive = list({g for ct in exclusive_genes for g in exclusive_genes[ct] if g in all_exclusive})
    filtered = {ct: [g for g in exclusive_genes[ct] if g in unique_exclusive] for ct in exclusive_genes}

    # All cross-type pairs
    mutually_exclusive_pairs = [
        (g1, g2)
        for ct1, ct2 in combinations(filtered.keys(), 2)
        for g1 in filtered[ct1]
        for g2 in filtered[ct2]
    ]
    return mutually_exclusive_pairs


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
