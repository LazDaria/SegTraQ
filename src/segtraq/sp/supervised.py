import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import squidpy as sq
from scipy import sparse
from scipy.stats import fisher_exact

from ..utils import _get_count_matrix, _get_genes, merge_into_obs, merge_into_uns


def mutually_exclusive_coexpression_rate(
    sdata,
    markers: dict[str, dict[str, list[str]]],
    tables_key: str = "table",
    tables_gene_key: str | None = None,
    tables_raw_counts_layer: str | None = None,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Assess co-expression of marker genes expected to be mutually exclusive.

    Candidate gene pairs are defined from positive and negative marker sets.
    For each pair, a one-sided Fisher's exact test evaluates whether the genes
    are detected together less frequently than expected under independence.

    Parameters
    ----------
    sdata : SpatialData-like
        Must contain `tables[tables_key]` as an AnnData with expression data.
    markers : dict
        Mapping of cell types to positive and negative markers:
        {cell_type: {"positive": list[str], "negative": list[str]}}.
    tables_key : str, optional, default="table"
        Key of the AnnData table in `sdata.tables`.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If None, `var_names` are used.
    tables_raw_counts_layer : str or None, optional
        Layer containing raw counts. If None, `adata.X` is used.
    inplace : bool, optional, default=True
        If True, store the resulting DataFrame in
        `sdata.tables[tables_key].uns["mutually_exclusive_coexpression_rate"]`.

    Returns
    -------
    pd.DataFrame
        One row per candidate marker-gene pair with columns:
        `gene1`, `gene2`, `odds_ratio`, `pvalue`, `a`, `b`, `c`, and `d`.

        Odds ratios below 1 indicate less co-expression than expected under
        independence. The one-sided Fisher p-value quantifies evidence for
        such mutual exclusivity.
    """
    adata = sdata.tables[tables_key]

    X = _get_count_matrix(adata, layer=tables_raw_counts_layer)
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    var_index = _get_genes(
        adata=adata,
        gene_key=tables_gene_key,
    )

    n_cells = X_dense.shape[0]

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
        # avoid requiring markers to be pre-filtered to genes present in the spatial data
        if g1 not in var_index or g2 not in var_index:
            continue
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

        # Fisher's exact test for under-co-occurrence (mutual exclusivity)
        try:
            odds_ratio, pval = fisher_exact(
                [[a, b], [c, d]],
                alternative="less",
            )
        except Exception:
            odds_ratio = np.nan
            pval = np.nan

        rows.append(
            {
                "gene1": g1,
                "gene2": g2,
                "odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else odds_ratio,
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
    tables_raw_counts_layer: str | None = None,
    tables_cell_id_key: str = "cell_id",
    tables_centroid_x_key: str = "x_centroid",
    tables_centroid_y_key: str = "y_centroid",
    tables_gene_key: str | None = None,
    require_neighbor_expression: bool = True,
    neighbors_key: str = "spatial_connectivities",
    inplace: bool = True,
):
    """
    Compute local negative-marker contamination per cell and per source-target
    cell-type pair.

    A gene is considered a locally relevant contamination marker for a target
    cell if it is:
    1. a negative marker of the target cell type, and
    2. a positive marker of at least one neighboring source cell type.

    If `require_neighbor_expression=True`, the gene must additionally be detected
    in at least one neighboring cell of the corresponding source cell type.

    Per-cell outputs (written to `adata.obs`):
        - contamination_counts:
            Total counts of locally relevant negative-marker genes in the focal cell.
        - contamination_strength:
            contamination_counts divided by the total transcript counts of the focal cell.
            This estimates the fraction of assigned transcripts that correspond to
            plausible local contamination.

    Source-target summaries (written to `adata.uns`):
        - contamination_matrix:
            Directed source-to-target matrix. Entry (c_src, c_tgt) is the fraction
            of evaluable target cells of type c_tgt that contain at least one locally
            relevant negative marker associated with source type c_src.
        - contamination_strength_matrix:
            Directed source-to-target matrix. Entry (c_src, c_tgt) is the mean
            source-specific contamination strength across evaluable target cells
            of type c_tgt with nonzero total counts.
        - contamination_evaluable_cells_matrix:
            Directed source-to-target matrix. Entry (c_src, c_tgt) is the number
            of target cells for which contamination from source type c_src could
            be evaluated.

    Returns
    -------
    per_cell_df : pandas.DataFrame
        Per-cell contamination counts and contamination strength.

    contamination_matrix_df : pandas.DataFrame
        Source-target matrix with the fraction of evaluable target cells contaminated
        by each source cell type.

    contamination_strength_matrix_df : pandas.DataFrame
        Source-target matrix with mean source-specific contamination strength.

    contamination_evaluable_cells_matrix_df : pandas.DataFrame
        Source-target matrix with the number of evaluable target cells per pair.
    """
    contamination_matrix_key = "contamination_matrix"
    contamination_strength_matrix_key = "contamination_strength_matrix"
    contamination_evaluable_cells_matrix_key = "contamination_evaluable_cells_matrix"

    # load expression matrix and metadata
    adata = sdata.tables[tables_key]

    X = _get_count_matrix(adata, layer=tables_raw_counts_layer)
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    var_index = _get_genes(adata=adata, gene_key=tables_gene_key)
    cell_types = np.asarray(adata.obs[cell_type_key])
    n_cells = X_dense.shape[0]
    total_counts = np.asarray(X_dense.sum(axis=1)).ravel()

    # compute neighborhood graph if missing
    if neighbors_key not in adata.obsp:
        warnings.warn(
            f"neighbors_key={neighbors_key} missing; computing Delaunay neighbors.",
            RuntimeWarning,
            stacklevel=2,
        )
        adata.obsm["spatial"] = adata.obs[[tables_centroid_x_key, tables_centroid_y_key]].to_numpy()
        sq.gr.spatial_neighbors_delaunay(adata)

    # extract neighbor indices
    G = adata.obsp[neighbors_key]
    if sparse.issparse(G):
        G = G.tocsr()
        neighbor_indices = [G[i].indices for i in range(n_cells)]
    else:
        G = np.asarray(G)
        neighbor_indices = [np.where(G[i] > 0)[0] for i in range(n_cells)]

    # marker sets
    positive_sets = {ct: set(m.get("positive", [])) for ct, m in markers.items()}
    negative_sets = {ct: set(m.get("negative", [])) for ct, m in markers.items()}

    all_cts = sorted({ct for ct in cell_types if not pd.isna(ct)})

    # precompute relevant genes for each directed source-target pair:
    # genes that are negative in the target and positive in the source
    type_pair_genes: dict[tuple[str, str], np.ndarray] = {}

    for c_tgt in all_cts:
        neg = negative_sets.get(c_tgt, set())

        for c_src in all_cts:
            genes = list(neg & positive_sets.get(c_src, set()))
            if not genes:
                continue

            idx = var_index.get_indexer(genes)
            idx = idx[idx >= 0]

            if idx.size:
                type_pair_genes[(c_src, c_tgt)] = idx

    # per-cell outputs
    contamination_counts = np.full(n_cells, np.nan, dtype=float)
    contamination_strength = np.full(n_cells, np.nan, dtype=float)

    # source-target accumulators
    pair_hit_cells = defaultdict(int)
    pair_evaluable_cells = defaultdict(int)
    pair_strength_sum = defaultdict(float)

    # iterate over target cells.
    for i, c_tgt in enumerate(cell_types):
        if pd.isna(c_tgt) or c_tgt not in negative_sets:
            continue

        if total_counts[i] == 0:
            continue

        nbs = neighbor_indices[i]
        if len(nbs) == 0:
            continue

        x_i = X_dense[i, :]
        nb_cts = cell_types[nbs]

        used_genes_cell = set()
        pair_counts_this_cell = defaultdict(float)

        # evaluate contamination separately for each neighboring source type
        for c_src in {ct for ct in nb_cts if not pd.isna(ct)}:
            pair = (c_src, c_tgt)
            gene_idx = type_pair_genes.get(pair)

            if gene_idx is None:
                continue

            nb_src = nbs[nb_cts == c_src]
            X_nb_src = X_dense[np.ix_(nb_src, gene_idx)]

            valid_gene_idx = []
            for k, g_idx in enumerate(gene_idx):
                # optionally require expression in at least one neighboring
                # source cell
                if require_neighbor_expression and not (X_nb_src[:, k] > 0).any():
                    continue
                valid_gene_idx.append(g_idx)

            if not valid_gene_idx:
                continue

            # at least one locally relevant marker exists, so the cell is evaluable
            if np.isnan(contamination_counts[i]):
                contamination_counts[i] = 0.0

            for g_idx in valid_gene_idx:
                x_i_g = x_i[g_idx]

                # source-specific counts are accumulated per source-target pair
                pair_counts_this_cell[pair] += x_i_g

                # per-cell counts should count each gene only once, even if the
                # gene is relevant for multiple neighboring source types
                if g_idx not in used_genes_cell:
                    contamination_counts[i] += x_i_g
                    used_genes_cell.add(g_idx)

        # per-cell contamination strength
        if not np.isnan(contamination_counts[i]):
            contamination_strength[i] = contamination_counts[i] / total_counts[i]

        # source-target summaries for this target cell
        for pair, counts in pair_counts_this_cell.items():
            pair_evaluable_cells[pair] += 1

            if counts > 0:
                pair_hit_cells[pair] += 1

            source_strength = counts / total_counts[i]
            pair_strength_sum[pair] += source_strength

    per_cell_df = pd.DataFrame(
        {
            tables_cell_id_key: adata.obs[tables_cell_id_key],
            "contamination_counts": contamination_counts,
            "contamination_strength": contamination_strength,
        }
    )

    # build source-target matrices
    idxmap = {ct: i for i, ct in enumerate(all_cts)}

    contamination_mat = np.full((len(all_cts), len(all_cts)), np.nan, dtype=float)
    strength_mat = np.full((len(all_cts), len(all_cts)), np.nan, dtype=float)
    evaluable_mat = np.full((len(all_cts), len(all_cts)), np.nan, dtype=float)

    for (c_src, c_tgt), n_eval in pair_evaluable_cells.items():
        row = idxmap[c_src]
        col = idxmap[c_tgt]

        evaluable_mat[row, col] = n_eval
        contamination_mat[row, col] = pair_hit_cells[(c_src, c_tgt)] / n_eval

    for (c_src, c_tgt), total_strength in pair_strength_sum.items():
        n = pair_evaluable_cells[(c_src, c_tgt)]
        strength_mat[idxmap[c_src], idxmap[c_tgt]] = total_strength / n

    contamination_matrix_df = pd.DataFrame(
        contamination_mat,
        index=all_cts,
        columns=all_cts,
    )

    contamination_strength_matrix_df = pd.DataFrame(
        strength_mat,
        index=all_cts,
        columns=all_cts,
    )

    contamination_evaluable_cells_matrix_df = pd.DataFrame(
        evaluable_mat,
        index=all_cts,
        columns=all_cts,
    )

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=per_cell_df,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )
        merge_into_uns(
            sdata,
            tables_key=tables_key,
            updates={
                contamination_matrix_key: contamination_matrix_df,
                contamination_strength_matrix_key: contamination_strength_matrix_df,
                contamination_evaluable_cells_matrix_key: contamination_evaluable_cells_matrix_df,
            },
        )

    return (
        per_cell_df,
        contamination_matrix_df,
        contamination_strength_matrix_df,
        contamination_evaluable_cells_matrix_df,
    )


def marker_purity(
    sdata,
    cell_type_key: str,
    markers: dict[str, dict[str, list[str]]],
    tables_key: str = "table",
    tables_raw_counts_layer: str | None = None,
    tables_cell_id_key: str = "cell_id",
    tables_centroid_x_key: str = "x_centroid",
    tables_centroid_y_key: str = "y_centroid",
    tables_gene_key: str | None = None,
    require_neighbor_expression: bool = True,
    neighbors_key: str = "spatial_connectivities",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute per-cell marker purity using balanced accuracy.

    For each cell of type c:
        - positive markers of c are expected to be expressed
        - relevant negative markers are negative markers of c that are also
          positive markers of neighboring cell types

    The score combines:
        - positive_marker_recall: fraction of expected positive markers expressed
          (analogous to recall/sensitivity at the marker-level)
        - negative_marker_avoidance: fraction of relevant negative markers avoided
          (analogous to specificity at the marker-level)
        - marker_balanced_accuracy: mean of sensitivity and specificity

    Parameters
    ----------
    sdata : SpatialData-like
        Must contain `tables[tables_key]` as an AnnData with expression and `.obs` metadata.
    cell_type_key : str
        Column in the AnnData `.obs` with cell-type labels.
    markers : dict
        A dictionary mapping cell types to their positive and negative markers, in the format
        {cell_type: {"positive": list[str], "negative": list[str]}}.
    tables_key : str, optional, default="table"
        Key of the AnnData table in `sdata.tables`.
    tables_raw_counts_layer : str | None, optional
        Layer containing count data. If `None`, `adata.X` is used if it looks
        like counts.
        If a layer is specified, it must exist and contain count-like values.
    tables_cell_id_key : str, optional, default="cell_id"
        Column in the AnnData `.obs` with unique cell IDs.
    tables_centroid_x_key : str or None, optional, default="x_centroid"
        Column in the cell table with the x-coordinate of the cell centroid.
    tables_centroid_y_key : str or None, optional, default="y_centroid"
        Column in the cell table with the y-coordinate of the cell centroid.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If `None`, `sdata.tables[tables_key].var_names` are used.
    require_neighbor_expression : bool, optional, default=True
        If True, contamination is only counted when the relevant gene is
        expressed in at least one neighboring cell of the source type.
    neighbors_key : str, optional, default="spatial_connectivities"
        Key in `adata.obsp` containing a cell x cell adjacency / connectivity
        matrix that defines the spatial neighborhood.
    inplace : bool, optional, default=True
        If True, store marker purity results in `sdata.tables[tables_key].obs`.

    Returns
    -------
    pandas.DataFrame
        Columns:
            - ``positive_marker_recall``
            - ``negative_marker_avoidance``
            - ``marker_balanced_accuracy``
            - ``n_evaluated_positive_markers``
            - ``n_evaluated_negative_markers``
    """
    adata = sdata.tables[tables_key]

    X = _get_count_matrix(adata, layer=tables_raw_counts_layer)
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    var_index = _get_genes(
        adata=adata,
        gene_key=tables_gene_key,
    )

    cell_types = np.asarray(adata.obs[cell_type_key])
    n_cells = X_dense.shape[0]

    # Compute spatial neighborhood graph if it is not already present.
    if neighbors_key not in adata.obsp:
        warnings.warn(
            f"neighbors_key='{neighbors_key}' not found in adata.obsp. "
            "A neighborhood graph based on Delaunay will be computed.",
            RuntimeWarning,
            stacklevel=2,
        )
        adata.obsm["spatial"] = adata.obs[[tables_centroid_x_key, tables_centroid_y_key]].to_numpy()
        sq.gr.spatial_neighbors_delaunay(adata)

    G = adata.obsp[neighbors_key]
    if sparse.issparse(G):
        G = G.tocsr()
        neighbor_indices = [G[i].indices for i in range(n_cells)]
    else:
        G = np.asarray(G)
        neighbor_indices = [np.where(G[i] > 0)[0] for i in range(n_cells)]

    # Keep only markers that are present in the spatial expression matrix.
    pos_sets = {ct: set(m.get("positive", [])) & set(var_index) for ct, m in markers.items()}
    neg_sets = {ct: set(m.get("negative", [])) & set(var_index) for ct, m in markers.items()}

    positive_recall = np.full(n_cells, np.nan, dtype=float)
    negative_avoidance = np.full(n_cells, np.nan, dtype=float)
    balanced_accuracy = np.full(n_cells, np.nan, dtype=float)
    n_pos_markers = np.zeros(n_cells, dtype=int)
    n_neg_markers = np.zeros(n_cells, dtype=int)

    for i, ct in enumerate(cell_types):
        if pd.isna(ct) or ct not in markers:
            continue

        pos_genes = pos_sets.get(ct, set())
        neg_all = neg_sets.get(ct, set())
        nbs = neighbor_indices[i]

        if not pos_genes:
            continue

        # Positive recall is computed globally for the focal cell type,
        # independent of the cell's neighborhood.
        pos_idx = var_index.get_indexer(list(pos_genes))
        pos_idx = pos_idx[pos_idx >= 0]

        n_pos_markers[i] = pos_idx.size
        pos_expr = X_dense[i, pos_idx] > 0
        positive_recall[i] = pos_expr.mean()

        # Negative avoidance is neighborhood-aware.
        if not neg_all or len(nbs) == 0:
            continue

        relevant_neg_genes = set()

        for nb_ct in pd.unique(cell_types[nbs]):
            if pd.isna(nb_ct) or nb_ct not in pos_sets:
                continue

            # Relevant negatives are focal negatives that are positive markers
            # of at least one neighboring cell type.
            candidate_genes = neg_all & pos_sets[nb_ct]
            if not candidate_genes:
                continue

            if not require_neighbor_expression:
                relevant_neg_genes.update(candidate_genes)
                continue

            nb_idx = nbs[cell_types[nbs] == nb_ct]

            # Optionally require the candidate gene to be expressed
            # in at least one neighbor of the corresponding source type.
            for g in candidate_genes:
                g_idx = var_index.get_loc(g)
                if (X_dense[nb_idx, g_idx] > 0).any():
                    relevant_neg_genes.add(g)

        if not relevant_neg_genes:
            continue

        neg_idx = var_index.get_indexer(list(relevant_neg_genes))
        neg_idx = neg_idx[neg_idx >= 0]

        n_neg_markers[i] = neg_idx.size
        neg_expr = X_dense[i, neg_idx] > 0

        negative_avoidance[i] = (~neg_expr).mean()
        balanced_accuracy[i] = 0.5 * (positive_recall[i] + negative_avoidance[i])

    result = pd.DataFrame(
        {
            tables_cell_id_key: adata.obs[tables_cell_id_key],
            "positive_marker_recall": positive_recall,
            "negative_marker_avoidance": negative_avoidance,
            "marker_balanced_accuracy": balanced_accuracy,
            "n_evaluated_positive_markers": n_pos_markers,
            "n_evaluated_negative_markers": n_neg_markers,
        },
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
