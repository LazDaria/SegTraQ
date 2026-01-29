import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from ..utils import _looks_like_counts, _score_negative_with_neighbors, _score_one_list, merge_into_obs
from .utils import add_neighbor_celltype_binary, assign_grid_splits, run_single_permutation


def compute_MECR(
    sdata,
    markers: dict[str, dict[str, list[str]]],
    tables_key: str = "table",
    pseudocount: float = 0.5,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute mutual exclusivity between marker genes using Fisher's exact test.
    Returns a DataFrame with one row per unordered gene pair.

    Parameters
    ----------
    sdata : SpatialData-like
        Must contain `tables[tables_key]` as an AnnData with expression and `.obs` metadata.
    markers : dict
        {cell_type: {"positive": list[str], "negative": list[str]}}.
    tables_key : str, optional, default="table"
        Key of the AnnData table in `sdata.tables`.
    pseudocount : float, optional, default=0.5
        Pseudocount added to all cells of the contingency table to avoid
        division by zero when computing odds ratios.
    inplace : bool, optional, default=True
        If True, store the resulting DataFrame in `sdata.tables[tables_key].uns["MECR"]`.

    Returns
    -------
    pd.DataFrame
        Columns:
            ['gene1', 'gene2', 'odds_ratio', 'pvalue', 'a', 'b', 'c', 'd']
        where (a, b, c, d) are the counts in the contingency table.
    """
    tbl = sdata.tables[tables_key]

    X = tbl.X
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    var_index = pd.Index(tbl.var_names)
    n_cells = arr.shape[0]
    pc = float(pseudocount)

    # --- build unique unordered candidate pairs ---
    candidate_pairs = set()
    pos_sets = {}

    for ct, d in markers.items():
        pos = set((d or {}).get("positive", []) or [])
        neg = set((d or {}).get("negative", []) or [])
        pos_sets[ct] = pos

        for g1 in pos:
            for g2 in neg:
                if g1 != g2:
                    a, b = (g1, g2) if g1 < g2 else (g2, g1)
                    candidate_pairs.add((a, b))

    # --- drop pairs co-positive in any cell type ---
    kept_pairs = [(g1, g2) for g1, g2 in candidate_pairs if not any(g1 in ps and g2 in ps for ps in pos_sets.values())]

    det = arr > 0

    rows = []

    for g1, g2 in kept_pairs:
        i1, i2 = var_index.get_loc(g1), var_index.get_loc(g2)
        e1, e2 = det[:, i1], det[:, i2]

        a = int((e1 & e2).sum())
        b = int((e1 & ~e2).sum())
        c = int((~e1 & e2).sum())
        d = n_cells - a - b - c

        # Fisher's exact returns exact p-value for under-co-occurrence (mutual exclusivity)
        try:
            _, pval = fisher_exact([[a, b], [c, d]], alternative="less")
        except Exception:
            pval = np.nan

        # odds ratio is computed with a pseudocount (Haldane–Anscombe correction)
        or_pc = ((a + pc) * (d + pc)) / ((b + pc) * (c + pc))

        rows.append(
            {
                "gene1": g1,
                "gene2": g2,
                "odds_ratio": float(or_pc),
                "pvalue": float(pval) if np.isfinite(pval) else np.nan,
                "a": a,
                "b": b,
                "c": c,
                "d": d,
            }
        )

    df = pd.DataFrame(rows)

    if inplace:
        tbl.uns["MECR"] = df

    return df


def calculate_neighbor_contamination(
    sdata,
    cell_type_key: str,
    markers: dict[str, dict[str, list[str]]],
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_centroid_x_key: str = "x_centroid",
    tables_centroid_y_key: str = "y_centroid",
    require_neighbor_expression: bool = True,
    neighbors_key: str = "spatial_connectivities",
    uns_key: str = "negative_marker_contamination",
    uns_key_binary: str = "negative_marker_contamination_binary",
    inplace: bool = True,
):
    """
    Compute per-cell negative-marker contamination and directed c_src→c_tgt
    contamination summaries.

    Per-cell outputs (written to .obs):
        - neg_marker_contam_counts:
            Total transcripts in the focal cell that belong to genes that are
            (i) negative markers of the focal cell type and
            (ii) positive markers of at least one neighboring cell type.
        - neg_marker_contam_fraction:
            For each such gene g, compute x_i(g) / (x_i(g) + mean_neighbor(g)),
            averaged across genes (neighbors pooled across all neighbor types).

    Type x type outputs (written to .uns):
        - uns_key:
            A directed matrix with entries = mean over genes of
            x_i(g) / (x_i(g) + mean_src(g)), aggregated across all contributing
            (cell, gene) observations for (c_src, c_tgt).
        - uns_key_binary:
            A directed matrix with entries =
            (# target cells of type c_tgt contaminated by c_src) / (# target cells of type c_tgt).
            Here “contaminated by c_src” is binary per target cell: at least one
            relevant gene (negative in target, positive in source) is detected in the
            target cell and detected in at least one neighbor of type c_src.

    Parameters
    ----------
    sdata : SpatialData-like
        Must contain `tables[tables_key]` as an AnnData with expression and `.obs` metadata.
    cell_type_key : str
        Column in the AnnData `.obs` with cell-type labels.
    markers : dict
        {cell_type: {"positive": list[str], "negative": list[str]}}.
    tables_key : str, optional, default="table"
        Key of the AnnData table in `sdata.tables`.
    tables_cell_id_key : str, optional, default="cell_id"
        Column in the AnnData `.obs` with unique cell IDs.
    tables_centroid_x_key : str or None, optional, default="x_centroid"
        Column in the cell table with the x-coordinate of the cell centroid.
    tables_centroid_y_key : str or None, optional, default="y_centroid"
        Column in the cell table with the y-coordinate of the cell centroid.
    require_neighbor_expression : bool, optional, default=True
        If True, contamination is only counted when the relevant gene is
        expressed in at least one neighboring cell of the source type.
    neighbors_key : str, optional, default="spatial_connectivities"
        Key in `adata.obsp` containing a cell x cell adjacency / connectivity
        matrix that defines the spatial neighborhood.
    uns_key : str, optional, default="negative_marker_contamination"
        Key in `.uns` under which the directed source → target mean contamination
        fraction matrix is stored.
    uns_key_binary : str, optional, default="negative_marker_contamination_binary"
        Key in `.uns` under which the directed source → target binary contamination
        proportion matrix is stored.
    inplace : bool, optional, default=True
        If True, store per-cell contamination metrics in
        `sdata.tables[tables_key].obs` and type-level matrices in `.uns`.

    Returns
    -------
    per_cell_df : pd.DataFrame
        Per-cell contamination metrics, indexed by cell ID.
    contam_matrix_df : pd.DataFrame
        Directed type x type mean contamination fraction matrix (c_src rows, c_tgt columns).
    contam_binary_df : pd.DataFrame
        Directed type x type binary contamination proportion matrix (c_src rows, c_tgt columns).
    """

    # ----------------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------------
    adata = sdata.tables[tables_key]
    X = adata.X
    genes = np.asarray(adata.var_names)
    var_index = pd.Index(genes)
    cell_types = np.asarray(adata.obs[cell_type_key])
    n_cells = X.shape[0]

    # Dense expression (counts)
    if _looks_like_counts(X):
        X_dense = X.toarray() if hasattr(X, "toarray") else X
    elif "raw" not in adata.layers:
        raise ValueError(
            f"'raw' layer does not exist in sdata.tables['{tables_key}'], "
            "and the main matrix does not look like counts."
        )
    else:
        raw = adata.layers["raw"]
        X_dense = raw.toarray() if hasattr(raw, "toarray") else raw

    # Neighbors
    if neighbors_key not in adata.obsp:
        warnings.warn(
            f"neighbors_key={neighbors_key} missing; computing Delaunay neighbors.",
            RuntimeWarning,
            stacklevel=2,
        )
        adata.obsm["spatial"] = adata.obs[[tables_centroid_x_key, tables_centroid_y_key]].to_numpy()
        import squidpy as sq  # local import so this function doesn't hard-require squidpy

        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")

    G = adata.obsp[neighbors_key]
    if sparse.issparse(G):
        G = G.tocsr()
        neighbor_indices = [G[i].indices for i in range(n_cells)]
    else:
        G = np.asarray(G)
        neighbor_indices = [np.where(G[i] > 0)[0] for i in range(n_cells)]

    # Marker sets
    pos_sets = {ct: set(m.get("positive", [])) for ct, m in markers.items()}
    neg_sets = {ct: set(m.get("negative", [])) for ct, m in markers.items()}

    # all cell types actually present
    all_cts = sorted({ct for ct in cell_types if not pd.isna(ct)})

    # denominator per target type for binary matrix
    tgt_totals = {ct: int(np.sum(cell_types == ct)) for ct in all_cts}

    # ----------------------------------------------------------------------
    # Precompute for each (c_src, c_tgt): negative(c_tgt) ∩ positive(c_src)
    # ----------------------------------------------------------------------
    type_pair_genes: dict[tuple[str, str], np.ndarray] = {}

    for c_tgt in all_cts:
        neg = neg_sets.get(c_tgt, set())
        if not neg:
            continue
        for c_src in all_cts:
            pos = pos_sets.get(c_src, set())
            genes_inter = list(neg & pos)
            if not genes_inter:
                continue
            idx = var_index.get_indexer(genes_inter)
            idx = idx[idx >= 0]
            if idx.size:
                type_pair_genes[(c_src, c_tgt)] = idx

    # ----------------------------------------------------------------------
    # Accumulators
    # ----------------------------------------------------------------------
    numer_cell = np.zeros(n_cells, dtype=float)
    sum_cell_frac = np.zeros(n_cells, dtype=float)
    count_cell_genes = np.zeros(n_cells, dtype=int)

    sum_pair = defaultdict(float)  # mean fraction numerator per (c_src, c_tgt)
    count_pair = defaultdict(int)  # gene contributions per (c_src, c_tgt)

    contam_cells_hit = defaultdict(int)  # target cells hit by source (binary)

    # ----------------------------------------------------------------------
    # Loop over cells
    # ----------------------------------------------------------------------
    for i in range(n_cells):
        c_tgt = cell_types[i]
        if pd.isna(c_tgt) or c_tgt not in neg_sets:
            continue

        nbs = neighbor_indices[i]
        if len(nbs) == 0:
            continue

        x_i = X_dense[i, :]
        nb_cts = cell_types[nbs]

        # track per-cell genes already counted (per-cell metrics)
        used_genes = set()

        # track which source types contaminate this target cell at least once
        contaminated_by_src = set()

        for c_src in {ct for ct in nb_cts if not pd.isna(ct)}:
            pair = (c_src, c_tgt)
            if pair not in type_pair_genes:
                continue

            gene_idx = type_pair_genes[pair]
            x_i_sub = x_i[gene_idx]

            # neighbors of this source type
            nb_src = nbs[nb_cts == c_src]
            if len(nb_src) == 0:
                continue

            X_nb_src = X_dense[np.ix_(nb_src, gene_idx)]

            # loop genes
            for k, g_idx in enumerate(gene_idx):
                x_i_g = x_i_sub[k]
                if x_i_g <= 0:
                    continue

                # only call it contamination if at least one neighbor expresses the gene
                x_nb_g = X_nb_src[:, k]
                if require_neighbor_expression:
                    mask_pos = x_nb_g > 0
                    if not mask_pos.any():
                        continue
                    mean_src = x_nb_g[mask_pos].mean()
                else:
                    mean_src = x_nb_g.mean()

                if not np.isfinite(mean_src):
                    continue

                # binary “this target cell is contaminated by this source type”
                contaminated_by_src.add(c_src)

                denom_all = x_i_g + mean_src
                if denom_all <= 0:
                    continue
                frac_g = x_i_g / denom_all

                # pair stats: always update (no used_genes gating)
                sum_pair[pair] += frac_g
                count_pair[pair] += 1

                # per-cell stats: update once per gene per cell
                if g_idx not in used_genes:
                    numer_cell[i] += x_i_g
                    sum_cell_frac[i] += frac_g
                    count_cell_genes[i] += 1
                    used_genes.add(g_idx)

        # update per-(c_src, c_tgt) binary hit counts once per cell
        for c_src in contaminated_by_src:
            contam_cells_hit[(c_src, c_tgt)] += 1

    # ----------------------------------------------------------------------
    # Build per-cell output
    # ----------------------------------------------------------------------
    contam_fraction = np.divide(
        sum_cell_frac,
        count_cell_genes,
        out=np.full(n_cells, np.nan),
        where=count_cell_genes > 0,
    )

    per_cell_df = pd.DataFrame(
        {
            "neg_marker_contam_counts": numer_cell,
            "neg_marker_contam_fraction": contam_fraction,
        },
        index=adata.obs[tables_cell_id_key],
    )

    # ----------------------------------------------------------------------
    # Build contamination matrices contam_matrix_df, contam_binary_df
    # ----------------------------------------------------------------------
    ct_list = all_cts
    idxmap = {ct: j for j, ct in enumerate(ct_list)}

    contam_mat = np.full((len(ct_list), len(ct_list)), np.nan, dtype=float)
    for (c_src, c_tgt), total in sum_pair.items():
        n = count_pair[(c_src, c_tgt)]
        if n > 0:
            contam_mat[idxmap[c_src], idxmap[c_tgt]] = total / n

    contam_matrix_df = pd.DataFrame(contam_mat, index=ct_list, columns=ct_list)

    # binary target-normalized matrix
    contam_bin = np.full((len(ct_list), len(ct_list)), np.nan, dtype=float)
    for (c_src, c_tgt), hits in contam_cells_hit.items():
        denom = tgt_totals.get(c_tgt, 0)
        if denom > 0:
            contam_bin[idxmap[c_src], idxmap[c_tgt]] = hits / denom
    contam_binary_df = pd.DataFrame(contam_bin, index=ct_list, columns=ct_list)

    # ----------------------------------------------------------------------
    # Write to .obs and .uns
    # ----------------------------------------------------------------------
    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=per_cell_df,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )
        sdata.tables[tables_key].uns[uns_key] = contam_matrix_df
        sdata.tables[tables_key].uns[uns_key_binary] = contam_binary_df

    return per_cell_df, contam_matrix_df, contam_binary_df


def calculate_marker_purity(
    sdata,
    cell_type_key: str,
    markers: dict[str, dict[str, list[str]]],
    use_quantiles: bool = False,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_centroid_x_key: str = "x_centroid",
    tables_centroid_y_key: str = "y_centroid",
    weight_cont: float = 0.7,
    require_neighbor_expression: bool = True,
    neighbors_key: str = "spatial_connectivities",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute per-cell marker purity using positive markers globally
    and negative markers restricted to the focal cell's neighborhood.

    Positive markers:
        - For each cell's annotated type c, use markers[c]["positive"].
        - Compute per-cell precision/recall/F1 relative to these markers,
          using either a quantile-based threshold or expression > 0.

    Negative markers (neighborhood-aware):
        - For a cell i of type c:
            * Find the cell types present in its spatial neighborhood,
              using `adata.obsp[neighbors_key]`.
            * Take c's negative markers: markers[c]["negative"].
            * Intersect them with the union of positive markers of the
              neighbor cell types.
            * Consider only genes that are present in at least one neighbor
              of the source type (if `require_neighbor_expression=True`).
            → "relevant negatives" for this cell.
        - Compute per-cell precision/recall/F1 for these relevant negatives.

    Finally:
        - Combine positive_F1 and negative_F1 into an overall F1_purity
          that rewards high positive-F1 and low negative-F1. Weighting
          factor weight_cont controls the importance of negative-F1.

    Parameters
    ----------
    sdata : SpatialData-like
        Must contain `tables[tables_key]` as an AnnData with expression and `.obs` metadata.
    cell_type_key : str
        Column in the AnnData `.obs` with cell-type labels.
    markers : dict
        {cell_type: {"positive": list[str], "negative": list[str]}}.
    use_quantiles : bool, optional, default=False
        If True, define predictions by the top-|markers| fraction per cell (rank-based);
        if False, use direct expression-based criteria (expression > 0).
    tables_key : str, optional, default="table"
        Key of the AnnData table in `sdata.tables`.
    tables_cell_id_key : str, optional, default="cell_id"
        Column in the AnnData `.obs` with unique cell IDs.
    tables_centroid_x_key : str or None, optional, default="x_centroid"
        Column in the cell table with the x-coordinate of the cell centroid.
    tables_centroid_y_key : str or None, optional, default="y_centroid"
        Column in the cell table with the y-coordinate of the cell centroid.
    require_neighbor_expression : bool, optional, default=True
        If True, contamination is only counted when the relevant gene is
        expressed in at least one neighboring cell of the source type.
    weight_cont : float, optional, default=0.7
        Weighting factor for negative marker F1 in the overall F1_purity.
        Higher values give more weight to negative F1. Must be in the range [0, 1].
    neighbors_key : str, optional, default="spatial_connectivities"
        Key in `adata.obsp` containing a cell x cell adjacency / connectivity
        matrix that defines the spatial neighborhood.
    inplace : bool, optional, default=True
        If True, store marker purity results in `sdata.tables[tables_key].obs`.

    Returns
    -------
    pandas.DataFrame
        Columns:
            [
             'positive_precision', 'positive_recall', 'positive_F1',
             'negative_precision', 'negative_recall', 'negative_F1',
             'F1_purity']
        indexed by cell ID.
    """
    if not (0 <= weight_cont <= 1):
        raise ValueError(f"weight_cont must be between 0 and 1 (inclusive), got {weight_cont}")

    adata = sdata.tables[tables_key]

    X = adata.X  # keep sparse if sparse

    if _looks_like_counts(X):
        X_dense = X.toarray() if hasattr(X, "toarray") else X
    elif "raw" not in adata.layers:
        raise ValueError(
            f"'raw' layer does not exist in sdata.tables['{tables_key}'], "
            "and the main matrix does not look like counts."
        )
    else:
        raw = adata.layers["raw"]
        X_dense = raw.toarray() if hasattr(raw, "toarray") else raw

    genes = np.asarray(adata.var_names)
    var_index = pd.Index(genes)
    cell_types = np.asarray(adata.obs[cell_type_key])
    n_cells, _n_genes = X.shape

    def _idx(lst) -> np.ndarray:
        if lst is None or len(lst) == 0:
            return np.empty(0, dtype=int)
        return var_index.get_indexer_for(lst)

    # Output arrays
    pos_prec = np.full(n_cells, np.nan, dtype=float)
    pos_rec = np.full(n_cells, np.nan, dtype=float)
    pos_f1 = np.full(n_cells, np.nan, dtype=float)

    neg_prec = np.full(n_cells, np.nan, dtype=float)
    neg_rec = np.full(n_cells, np.nan, dtype=float)
    neg_f1 = np.full(n_cells, np.nan, dtype=float)

    purity = np.full(n_cells, np.nan, dtype=float)

    # ---------------- POSITIVE markers (per type, global) -------------------
    unique_cts = pd.unique(cell_types)

    for ct in unique_cts:
        mask_cells = cell_types == ct
        idx_cells = np.where(mask_cells)[0]
        if idx_cells.size == 0:
            continue

        # missing/unknown type → leave NaNs
        if pd.isna(ct) or ct not in markers:
            continue

        m = markers[ct]
        pos_idx = _idx(m.get("positive", []))
        neg_idx = _idx(m.get("negative", []))
        pos_idx = pos_idx[pos_idx >= 0]
        neg_idx = neg_idx[neg_idx >= 0]
        all_idx = np.unique(np.concatenate([pos_idx, neg_idx])) if neg_idx.size else pos_idx

        # If no pos markers, nothing to compute
        if pos_idx.size == 0:
            continue

        X_ct = X_dense[mask_cells]

        p_prec_ct, p_rec_ct, p_f1_ct = _score_one_list(
            X_ct,
            pos_idx,
            all_idx,
            use_quantiles=use_quantiles,
        )
        pos_prec[idx_cells] = p_prec_ct
        pos_rec[idx_cells] = p_rec_ct
        pos_f1[idx_cells] = p_f1_ct

    # ------------- NEGATIVE markers: neighborhood-aware ---------------------
    if neighbors_key not in adata.obsp:
        warnings.warn(
            f"neighbors_key='{neighbors_key}' not found in adata.obsp. "
            "A neighborhood graph based on Delaunay will be computed.",
            RuntimeWarning,
            stacklevel=2,
        )
        sdata[tables_key].obsm["spatial"] = (
            sdata[tables_key].obs[[tables_centroid_x_key, tables_centroid_y_key]].to_numpy()
        )
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")

    G = adata.obsp[neighbors_key]

    if sparse.issparse(G):
        G = G.tocsr()
        neighbor_indices = [G[i].indices for i in range(n_cells)]
    else:
        G = np.asarray(G)
        neighbor_indices = [np.where(G[i] > 0)[0] for i in range(n_cells)]

    neg_prec, neg_rec, neg_f1 = _score_negative_with_neighbors(
        X_dense=X_dense,
        cell_types=cell_types,
        markers=markers,
        genes=genes,
        require_neighbor_expression=require_neighbor_expression,
        neighbor_indices=neighbor_indices,
        use_quantiles=use_quantiles,
    )

    # ------------------------ F1_purity -------------------------------------
    p_f1_all = pos_f1
    n_f1_all = neg_f1

    mask_valid = ~np.isnan(p_f1_all) & ~np.isnan(n_f1_all)
    if np.any(mask_valid):
        p = p_f1_all[mask_valid]
        n = 1 - n_f1_all[mask_valid]
        weight_id = 1 - weight_cont
        purity_vals = weight_cont * n + weight_id * p
        purity[mask_valid] = purity_vals
    # cells without pos or neg markers stay NaN for purity

    # Build DataFrame
    result = pd.DataFrame(
        {
            "positive_precision": pos_prec,
            "positive_recall": pos_rec,
            "positive_F1": pos_f1,
            "negative_precision": neg_prec,
            "negative_recall": neg_rec,
            "negative_F1": neg_f1,
            "F1_purity": purity,
        },
        index=adata.obs[tables_cell_id_key],
    )

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=result,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )

    return result


def neighbor_prediction(
    sdata,
    ct1,
    ct2,
    tables_key="table",
    cell_type_key="transferred_cell_type",
    tables_x_key="x_centroid",
    tables_y_key="y_centroid",
    grid_shape=(10, 10),
    n_permutations=100,
    seed=0,
    inplace=True,
    n_jobs=-1,
):
    """
    See if it is possible to predict adjacency of ct1 next to ct2 based on ct1's expression profiles using
    logistic regression.
    Parameters
    ----------
    sdata : object
        Container object expected to expose a mapping-like attribute `tables` such
        that `sdata.tables[tables_key]` returns an AnnData-like object. The AnnData
        must provide:
          - .obs: a pandas.DataFrame containing at least the `cell_type_key` column
          - .var_names: iterable of gene names
          - .X: expression matrix (numpy array or sparse matrix)
          - .obsm: a dict-like object; the function will look for or add
            "neighbor_celltype_binary"
    ct1 : str or int
        The focal cell type (as stored in `.obs[cell_type_key]`) for which to
        predict the presence of neighboring ct2 cells.
    ct2 : str or int
        The neighbor cell type (column name in `adata.obsm["neighbor_celltype_binary"]`)
        whose presence/absence among neighbors is the target binary outcome.
    tables_key : str, optional
        Key in `sdata.tables` to read the AnnData from (default: "table").
    cell_type_key : str, optional
        Column name in `.obs` containing transferred/annotated cell types
        (default: "transferred_cell_type").
    tables_x_key, tables_y_key : str, optional
        Column names in `.obs` used as spatial coordinates for grid splitting and
        for computing neighbor relations if `neighbor_celltype_binary` must be added.
        (default: "x_centroid", "y_centroid").
    grid_shape : tuple of two ints, optional
        Number of spatial grid cells along (nx, ny) used to assign spatial folds;
        cells within the same grid tile go to the same split (default: (10, 10)).
    n_permutations : int, optional
        Number of permutations to build an empirical null distribution of AP scores
        (default: 100).
    seed : int, optional
        Random seed used for reproducible splitting and permutation seeds (default: 0).
    inplace : bool, optional
        If True, operate on the AnnData in `sdata.tables[tables_key]` directly and
        write results into its `.uns`. If False, operate on a copy and return the
        result without mutating the original (default: True).
    n_jobs : int, optional
        Number of parallel jobs for permutation computation. Passed to joblib. If
        -1, use all available cores. The implementation uses the 'threading'
        backend to reduce memory-copy overhead (default: -1).
    Returns
    -------
    dict
        A dictionary describing the trained model and permutation results, saved
        into `adata.uns["neighbor_prediction"]` and also returned. Keys include:
          - "model_params": {"weights": array, "intercept": float}
              Coefficients of the trained high-precision logistic regression.
          - "test_ap": float
              Observed average-precision (AP) on the spatial test fold.
          - "empirical_p_value": float
              Fraction of permutation APs greater than or equal to observed AP.
          - "null_aps": ndarray, shape (n_permutations,)
              AP scores computed under the null (permuted) models.
          - "gene_names": ndarray
              Names of features (genes) corresponding to model coefficients.
          - "splits": {"train_indices": array, "test_indices": array}
              Indices (into the focal-cell index set) of the train/test split.
    Raises
    ------
    ValueError
        If fewer than 10 cells of type `ct1` are available (insufficient data).
    """
    if inplace:
        adata = sdata.tables[tables_key]
    else:
        adata = sdata.tables[tables_key].copy()

    # --- 1. Data Preparation ---
    # here, we add a matrix of shape (cells, cell_types) in
    # adata.obsm["neighbor_celltype_binary"] that indicates if a cell has
    # at least one neighbor of that cell type
    if "neighbor_celltype_binary" not in adata.obsm:
        add_neighbor_celltype_binary(
            adata, cell_type_col=cell_type_key, tables_x_key=tables_x_key, tables_y_key=tables_y_key
        )

    mask_focal = adata.obs[cell_type_key].astype(str) == str(ct1)
    idx_focal = np.where(mask_focal)[0]

    if len(idx_focal) < 10:
        raise ValueError(f"Too few cells ({len(idx_focal)}) of type {ct1}.")

    y_all = adata.obsm["neighbor_celltype_binary"].iloc[idx_focal][ct2].astype(int).values

    # check if there are both positive and negative samples
    if np.unique(y_all).size < 2:
        raise ValueError(
            f"Could not find both positive and negative samples for cell type {ct2} among neighbors of {ct1}."
        )

    # Check for log normalization
    if _looks_like_counts(adata.X):
        warnings.warn(
            "Reference adata does not appear log-normalized."
            "Counts will be log1p-transformed before running label transfer."
            "Raw counts will be stored in `adata.layers['raw']`.",
            RuntimeWarning,
            stacklevel=2,
        )
        adata.layers["raw"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    X_all = adata.X[idx_focal]
    if sparse.issparse(X_all):
        X_all = X_all.toarray()

    gene_names = np.array(adata.var_names)

    # --- 2. Splitting & Standardization ---
    # to make sure there is as little spatial leakage as possible
    # between train and test, we assign splits based on a spatial grid
    is_train, is_test = assign_grid_splits(
        adata, mask_focal, grid_shape, seed=seed, tables_x_key=tables_x_key, tables_y_key=tables_y_key
    )

    scaler = StandardScaler()
    scaler.fit(X_all[is_train])
    X_train = scaler.transform(X_all[is_train])
    X_test = scaler.transform(X_all[is_test])
    y_train = y_all[is_train]
    y_test = y_all[is_test]

    # check that both classes are present in training data
    if np.unique(y_train).size < 2:
        raise ValueError(f"Training data after spatial split does not contain both classes for cell type {ct2}.")
    if np.unique(y_test).size < 2:
        raise ValueError(f"Test data after spatial split does not contain both classes for cell type {ct2}.")

    # --- 3. Model Fitting ---
    rng = np.random.RandomState(seed)
    perm_scores = np.zeros(n_permutations)

    # Parameters for the logistic regression
    real_model_params = {
        "solver": "liblinear",
        "penalty": "l2",
        "C": 1.0,
        "class_weight": "balanced",
        "tol": 1e-4,
        "max_iter": 1000,
        "random_state": seed,
    }

    real_model = LogisticRegression(**real_model_params)
    real_model.fit(X_train, y_train)

    # Calculate observed score
    obs_probs = real_model.predict_proba(X_test)[:, 1]
    observed_score = average_precision_score(y_test, obs_probs)

    weights = real_model.coef_.ravel()
    intercept = real_model.intercept_[0]

    # --- 4. Permutation Testing ---
    # We use the 'threading' backend to avoid memory copy/serialization overhead.
    # We generate a unique seed for each permutation for better statistical validity.
    seeds = [rng.randint(2**32 - 1) for _ in range(n_permutations)]

    perm_scores = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(run_single_permutation)(obs_probs, y_test, seed=s) for s in seeds
    )
    perm_scores = np.array(perm_scores)
    p_value = np.mean(perm_scores >= observed_score)

    adata.uns["neighbor_prediction"] = {
        "model_params": {"weights": weights, "intercept": intercept},
        "test_ap": observed_score,
        "empirical_p_value": p_value,
        "null_aps": perm_scores,
        "gene_names": gene_names,
        "splits": {"train_indices": idx_focal[is_train], "test_indices": idx_focal[is_test]},
    }

    return adata.uns["neighbor_prediction"]
