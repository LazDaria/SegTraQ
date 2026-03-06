import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import squidpy as sq
from scipy import sparse
from scipy.stats import fisher_exact

from ..utils import merge_into_obs
from .utils import _get_count_matrix, _score_neighbor_negative_markers, _score_marker_detection


def mutually_exclusive_coexpression_rate(
    sdata,
    markers: dict[str, dict[str, list[str]]],
    tables_key: str = "table",
    layer: str | None = None,
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
        A dictionary mapping cell types to their positive and negative markers, in the format
        {cell_type: {"positive": list[str], "negative": list[str]}}.
    tables_key : str, optional, default="table"
        Key of the AnnData table in `sdata.tables`.
    layer : str | None, optional
        Layer containing count data. If `None`, `adata.X` is used if it looks
        like counts, otherwise `adata.layers["counts"]` is used if available.
        If a layer is specified, it must exist and contain count-like values.
    pseudocount : float, optional, default=0.5
        Pseudocount added to all cells of the contingency table to avoid
        division by zero when computing odds ratios.
        This is equivalent to the Haldane-Anscombe correction, see https://pmc.ncbi.nlm.nih.gov/articles/PMC7398076.
    inplace : bool, optional, default=True
        If True, store the resulting DataFrame in `sdata.tables[tables_key].uns["MECR"]`.

    Returns
    -------
    pd.DataFrame
        Columns:
            ['gene1', 'gene2', 'odds_ratio', 'pvalue', 'a', 'b', 'c', 'd']
        where (a, b, c, d) are the counts in the contingency table.
    """
    adata = sdata.tables[tables_key]

    X = _get_count_matrix(adata, layer=layer, tables_key=tables_key)
    X_dense = X.toarray() if hasattr(X, "toarray") else X

    var_index = pd.Index(adata.var_names)
    n_cells = X_dense.shape[0]
    pseudocount = float(pseudocount)

    # --- build unique unordered candidate pairs ---
    candidate_pairs = set()
    positive_sets = {}

    # go through all reference markers (positive and negative) for a given cell type
    for ct, d in markers.items():
        positive = set((d or {}).get("positive", []) or [])
        negative = set((d or {}).get("negative", []) or [])
        # record the positive markers for the given cell type
        positive_sets[ct] = positive

        for pos in positive:
            for neg in negative:
                # only consider pairs of different genes
                if pos != neg:
                    # order the candidates genes alphabetically
                    a, b = (pos, neg) if pos < neg else (neg, pos)
                    candidate_pairs.add((a, b))

    # --- drop pairs co-positive in any cell type ---
    kept_pairs = [
        (pos, neg) for pos, neg in candidate_pairs if not any(pos in ps and neg in ps for ps in positive_sets.values())
    ]

    # only consider positive expression values in the spatial data
    det = X_dense > 0

    rows = []
    # go over all positive/negative gene pairs from the scRNAseq reference that were kept through the filtering
    for g1, g2 in kept_pairs:
        i1, i2 = var_index.get_loc(g1), var_index.get_loc(g2)
        # extract all cell types positive/negative for combination from the spatial data for the
        # mutually exclusive gene pairs found in the scRNA seq data.
        e1, e2 = det[:, i1], det[:, i2]
        # count all the occurences of the confusion table
        a = int((e1 & e2).sum())
        b = int((e1 & ~e2).sum())
        c = int((~e1 & e2).sum())
        d = int((~e1 & ~e2).sum())
        assert d == n_cells - a - b - c, (
            "Contingency table counts do not sum to total number of cells. "
            "Please report this to the developers of SegTraQ."
        )

        # Fisher's exact returns exact p-value for under-co-occurrence (mutual exclusivity)
        try:
            _, pval = fisher_exact([[a, b], [c, d]], alternative="less")
        except Exception:
            pval = np.nan

        # odds ratio is computed with a pseudocount (Haldane–Anscombe correction)
        or_pseudocount = ((a + pseudocount) * (d + pseudocount)) / ((b + pseudocount) * (c + pseudocount))

        rows.append(
            {
                "gene1": g1,
                "gene2": g2,
                "odds_ratio": float(or_pseudocount),
                "pvalue": float(pval) if np.isfinite(pval) else np.nan,
                "a": a,
                "b": b,
                "c": c,
                "d": d,
            }
        )

    df = pd.DataFrame(rows)

    if inplace:
        adata.uns["mutually_exclusive_coexpression_rate"] = df

    return df


def neighbor_contamination(
    sdata,
    cell_type_key: str,
    markers: dict[str, dict[str, list[str]]],
    tables_key: str = "table",
    layer: str | None = None,
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
        - negative_marker_contamination_counts:
            Total transcripts in the focal cell that belong to genes that are
            (i) negative markers of the focal cell type and
            (ii) positive markers of at least one neighboring cell type.
        - negative_marker_contamination_fraction:
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
    layer : str | None, optional
        Layer containing count data. If `None`, `adata.X` is used if it looks
        like counts, otherwise `adata.layers["counts"]` is used if available.
        If a layer is specified, it must exist and contain count-like values.
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
    contamination_matrix_df : pd.DataFrame
        Directed type x type mean contamination fraction matrix (c_src rows, c_tgt columns).
        This matrix report the average strength of contamination from one cell type into another.
    contamination_binary_df : pd.DataFrame
        Directed type x type binary contamination proportion matrix (c_src rows, c_tgt columns).
        This matrix reports what percentage of cells of a target type is contaminated by each source type.
    """

    # ----------------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------------
    adata = sdata.tables[tables_key]
    X = _get_count_matrix(adata, layer=layer, tables_key=tables_key)
    X_dense = X.toarray() if hasattr(X, "toarray") else X

    genes = np.asarray(adata.var_names)
    var_index = pd.Index(genes)
    cell_types = np.asarray(adata.obs[cell_type_key])
    n_cells = X.shape[0]

    # Checking if neighborhood graph is present, else compute Delaunay triangulation
    if neighbors_key not in adata.obsp:
        warnings.warn(
            f"neighbors_key={neighbors_key} missing; computing Delaunay neighbors.",
            RuntimeWarning,
            stacklevel=2,
        )
        adata.obsm["spatial"] = adata.obs[[tables_centroid_x_key, tables_centroid_y_key]].to_numpy()

        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")
    # extract indices from the neighborhood graph
    G = adata.obsp[neighbors_key]
    if sparse.issparse(G):
        G = G.tocsr()
        neighbor_indices = [G[i].indices for i in range(n_cells)]
    else:
        G = np.asarray(G)
        neighbor_indices = [np.where(G[i] > 0)[0] for i in range(n_cells)]

    # Marker sets
    positive_sets = {ct: set(m.get("positive", [])) for ct, m in markers.items()}
    negative_sets = {ct: set(m.get("negative", [])) for ct, m in markers.items()}

    # get the set of all cell types present in the anndata object
    all_cts = sorted({ct for ct in cell_types if not pd.isna(ct)})

    # count the occurences of the cell types
    tgt_totals = {ct: int(np.sum(cell_types == ct)) for ct in all_cts}

    # ----------------------------------------------------------------------
    # Precompute for each (c_src, c_tgt): negative(c_tgt) ∩ positive(c_src)
    # ----------------------------------------------------------------------
    type_pair_genes: dict[tuple[str, str], np.ndarray] = {}

    for c_tgt in all_cts:
        neg = negative_sets.get(c_tgt, set())
        if not neg:
            continue
        for c_src in all_cts:
            pos = positive_sets.get(c_src, set())
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
    number_cell = np.zeros(n_cells, dtype=float)
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
        if pd.isna(c_tgt) or c_tgt not in negative_sets:
            continue

        nbs = neighbor_indices[i]
        if len(nbs) == 0:
            continue

        x_i = X_dense[i, :]
        # get cells around the target cell
        nb_cts = cell_types[nbs]

        # track per-cell genes already counted (per-cell metrics)
        used_genes = set()

        # track which source types contaminate this target cell at least once
        contaminated_by_src = set()

        for c_src in {ct for ct in nb_cts if not pd.isna(ct)}:
            pair = (c_src, c_tgt)
            if pair not in type_pair_genes:
                continue
            # get gene indices for cell type pair
            gene_idx = type_pair_genes[pair]
            # get gene expression for this pair
            x_i_sub = x_i[gene_idx]

            # neighbors of this source type
            nb_src = nbs[nb_cts == c_src]
            if len(nb_src) == 0:
                continue
            # get gene expression for the neighbours
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
                    number_cell[i] += x_i_g
                    sum_cell_frac[i] += frac_g
                    count_cell_genes[i] += 1
                    used_genes.add(g_idx)

        # update per-(c_src, c_tgt) binary hit counts once per cell
        for c_src in contaminated_by_src:
            contam_cells_hit[(c_src, c_tgt)] += 1

    # ----------------------------------------------------------------------
    # Build per-cell output
    # ----------------------------------------------------------------------
    contamination_fraction = np.divide(
        sum_cell_frac,
        count_cell_genes,
        out=np.full(n_cells, np.nan),
        where=count_cell_genes > 0,
    )

    per_cell_df = pd.DataFrame(
        {
            "negative_marker_contamination_counts": number_cell,
            "negative_marker_contamination_fraction": contamination_fraction,
        },
        index=adata.obs[tables_cell_id_key],
    )

    # ----------------------------------------------------------------------
    # Build contamination matrices contamination_matrix_df, contamination_binary_df
    # ----------------------------------------------------------------------
    ct_list = all_cts
    idxmap = {ct: j for j, ct in enumerate(ct_list)}

    contamination_mat = np.full((len(ct_list), len(ct_list)), np.nan, dtype=float)
    for (c_src, c_tgt), total in sum_pair.items():
        n = count_pair[(c_src, c_tgt)]
        if n > 0:
            contamination_mat[idxmap[c_src], idxmap[c_tgt]] = total / n

    contamination_matrix_df = pd.DataFrame(contamination_mat, index=ct_list, columns=ct_list)

    # binary target-normalized matrix
    contam_bin = np.full((len(ct_list), len(ct_list)), np.nan, dtype=float)
    for (c_src, c_tgt), hits in contam_cells_hit.items():
        denom = tgt_totals.get(c_tgt, 0)
        if denom > 0:
            contam_bin[idxmap[c_src], idxmap[c_tgt]] = hits / denom
    contamination_binary_df = pd.DataFrame(contam_bin, index=ct_list, columns=ct_list)

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
        sdata.tables[tables_key].uns[uns_key] = contamination_matrix_df
        sdata.tables[tables_key].uns[uns_key_binary] = contamination_binary_df

    return per_cell_df, contamination_matrix_df, contamination_binary_df


def marker_purity(
    sdata,
    cell_type_key: str,
    markers: dict[str, dict[str, list[str]]],
    use_quantiles: bool = False,
    tables_key: str = "table",
    layer: str | None = None,
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
        A dictionary mapping cell types to their positive and negative markers, in the format
        {cell_type: {"positive": list[str], "negative": list[str]}}.
    use_quantiles : bool, optional, default=False
        If True, define predictions by the top-|markers| fraction per cell (rank-based);
        if False, use direct expression-based criteria (expression > 0).
    tables_key : str, optional, default="table"
        Key of the AnnData table in `sdata.tables`.
    layer : str | None, optional
        Layer containing count data. If `None`, `adata.X` is used if it looks
        like counts, otherwise `adata.layers["counts"]` is used if available.
        If a layer is specified, it must exist and contain count-like values.
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

    # TODO: keep sparse if sparse
    X = _get_count_matrix(adata, layer=layer, tables_key=tables_key)
    X_dense = X.toarray() if hasattr(X, "toarray") else X

    genes = np.asarray(adata.var_names)
    var_index = pd.Index(genes)
    cell_types = np.asarray(adata.obs[cell_type_key])
    n_cells, _n_genes = X.shape

    def _idx(lst) -> np.ndarray:
        if lst is None or len(lst) == 0:
            return np.empty(0, dtype=int)
        return var_index.get_indexer_for(lst)

    # Output arrays initialisation
    pos_precision = np.full(n_cells, np.nan, dtype=float)
    pos_recall = np.full(n_cells, np.nan, dtype=float)
    pos_f1 = np.full(n_cells, np.nan, dtype=float)

    neg_precision = np.full(n_cells, np.nan, dtype=float)
    neg_recall = np.full(n_cells, np.nan, dtype=float)
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

        p_precision_ct, p_recall_ct, p_f1_ct = _score_marker_detection(
            X_ct,
            pos_idx,
            all_idx,
            use_quantiles=use_quantiles,
        )
        pos_precision[idx_cells] = p_precision_ct
        pos_recall[idx_cells] = p_recall_ct
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

    neg_precision, neg_recall, neg_f1 = _score_neighbor_negative_markers(
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
            "positive_precision": pos_precision,
            "positive_recall": pos_recall,
            "positive_F1": pos_f1,
            "negative_precision": neg_precision,
            "negative_recall": neg_recall,
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
