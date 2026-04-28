import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import squidpy as sq
from scipy import sparse
from scipy.stats import fisher_exact

from ..utils import merge_into_obs
from .utils import _get_count_matrix


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
    number_cell = np.full(n_cells, np.nan, dtype=float)
    sum_cell_frac = np.zeros(n_cells, dtype=float)
    count_cell_genes = np.zeros(n_cells, dtype=int)
    evaluable_cell = np.zeros(n_cells, dtype=bool)

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

        # mark cell as evaluable if at least one relevant local negative marker exists
        cell_has_relevant_gene = False

        for c_src in {ct for ct in nb_cts if not pd.isna(ct)}:
            pair = (c_src, c_tgt)
            if pair not in type_pair_genes:
                continue
            # get gene indices for cell type pair
            gene_idx = type_pair_genes[pair]

            # neighbors of this source type
            nb_src = nbs[nb_cts == c_src]
            if len(nb_src) == 0:
                continue
            # get gene expression for the neighbours
            X_nb_src = X_dense[np.ix_(nb_src, gene_idx)]

            # loop genes
            for k, g_idx in enumerate(gene_idx):
                x_nb_g = X_nb_src[:, k]
                if require_neighbor_expression:
                    mask_pos = x_nb_g > 0
                    if not mask_pos.any():
                        continue
                    mean_src = x_nb_g[mask_pos].mean()
                else:
                    mean_src = x_nb_g.mean()

                if not np.isfinite(mean_src) or mean_src <= 0:
                    continue

                # at least one valid local negative marker exists for this cell
                cell_has_relevant_gene = True

                x_i_g = x_i[g_idx]
                denom_all = x_i_g + mean_src
                if denom_all <= 0:
                    continue
                frac_g = x_i_g / denom_all

                # pair stats: always update for evaluable genes, including clean zeros
                sum_pair[pair] += frac_g
                count_pair[pair] += 1

                # per-cell stats: update once per gene per cell
                if g_idx not in used_genes:
                    number_cell[i] = 0.0 if np.isnan(number_cell[i]) else number_cell[i]
                    number_cell[i] += x_i_g
                    sum_cell_frac[i] += frac_g
                    count_cell_genes[i] += 1
                    used_genes.add(g_idx)

                # binary “this target cell is contaminated by this source type”
                if x_i_g > 0:
                    contaminated_by_src.add(c_src)

        evaluable_cell[i] = cell_has_relevant_gene
        if evaluable_cell[i] and np.isnan(number_cell[i]):
            number_cell[i] = 0.0

        # update per-(c_src, c_tgt) binary hit counts once per cell
        for c_src in contaminated_by_src:
            contam_cells_hit[(c_src, c_tgt)] += 1

    # ----------------------------------------------------------------------
    # Build per-cell output
    # ----------------------------------------------------------------------
    contamination_fraction = np.full(n_cells, np.nan, dtype=float)
    mask_eval = evaluable_cell & (count_cell_genes > 0)
    contamination_fraction[mask_eval] = sum_cell_frac[mask_eval] / count_cell_genes[mask_eval]
    contamination_fraction[evaluable_cell & (count_cell_genes == 0)] = 0.0

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
    tables_key: str = "table",
    layer: str | None = None,
    tables_cell_id_key: str = "cell_id",
    tables_centroid_x_key: str = "x_centroid",
    tables_centroid_y_key: str = "y_centroid",
    require_neighbor_expression: bool = True,
    neighbors_key: str = "spatial_connectivities",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute per-cell marker purity using balanced accuracy.

    For each cell of type c:
        - positive markers of c are expected to be expressed.
        - relevant negative markers are negative markers of c that are also
          positive markers of neighboring cell types.

    The score combines:
        - positive_marker_recall: sensitivity, i.e. fraction of positive markers expressed.
        - negative_marker_avoidance: specificity, i.e. fraction of relevant negative markers avoided.
        - marker_balanced_accuracy: mean of sensitivity and specificity.

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
    layer : str | None, optional
        Layer containing count data. If `None`, `adata.X` is used if it looks
        like counts, otherwise `adata.layers["counts"]` is used if available.
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
    inplace : bool, optional, default=True
        If True, store marker purity results in `sdata.tables[tables_key].obs`.

    Returns
    -------
    pandas.DataFrame
        Columns:
            [
             'positive_marker_recall',
             'negative_marker_avoidance',
             'marker_balanced_accuracy'
            ]
        indexed by cell ID.
    """
    adata = sdata.tables[tables_key]

    X = _get_count_matrix(adata, layer=layer, tables_key=tables_key)
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    genes = np.asarray(adata.var_names)
    var_index = pd.Index(genes)
    valid_genes = set(var_index)

    cell_types = np.asarray(adata.obs[cell_type_key])
    n_cells = X_dense.shape[0]

    if neighbors_key not in adata.obsp:
        warnings.warn(
            f"neighbors_key='{neighbors_key}' not found in adata.obsp. "
            "A neighborhood graph based on Delaunay will be computed.",
            RuntimeWarning,
            stacklevel=2,
        )
        adata.obsm["spatial"] = adata.obs[[tables_centroid_x_key, tables_centroid_y_key]].to_numpy()
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")

    G = adata.obsp[neighbors_key]
    if sparse.issparse(G):
        G = G.tocsr()
        neighbor_indices = [G[i].indices for i in range(n_cells)]
    else:
        G = np.asarray(G)
        neighbor_indices = [np.where(G[i] > 0)[0] for i in range(n_cells)]

    pos_sets = {ct: set(m.get("positive", [])) & valid_genes for ct, m in markers.items()}
    neg_sets = {ct: set(m.get("negative", [])) & valid_genes for ct, m in markers.items()}

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

        if not pos_genes or not neg_all or len(nbs) == 0:
            continue

        pos_idx = var_index.get_indexer(list(pos_genes))
        n_pos_markers[i] = pos_idx.size
        pos_expr = X_dense[i, pos_idx] > 0
        positive_recall[i] = pos_expr.mean()

        relevant_neg_genes = set()

        for nb_ct in pd.unique(cell_types[nbs]):
            if pd.isna(nb_ct) or nb_ct not in pos_sets:
                continue

            candidate_genes = neg_all & pos_sets[nb_ct]
            if not candidate_genes:
                continue

            if not require_neighbor_expression:
                relevant_neg_genes.update(candidate_genes)
                continue

            nb_idx = nbs[cell_types[nbs] == nb_ct]

            for g in candidate_genes:
                g_idx = var_index.get_loc(g)
                if (X_dense[nb_idx, g_idx] > 0).any():
                    relevant_neg_genes.add(g)

        if not relevant_neg_genes:
            continue

        neg_idx = var_index.get_indexer(list(relevant_neg_genes))
        n_neg_markers[i] = neg_idx.size

        neg_expr = X_dense[i, neg_idx] > 0
        negative_avoidance[i] = (~neg_expr).mean()
        balanced_accuracy[i] = 0.5 * (positive_recall[i] + negative_avoidance[i])

    result = pd.DataFrame(
        {
            "positive_marker_recall": positive_recall,
            "negative_marker_avoidance": negative_avoidance,
            "marker_balanced_accuracy": balanced_accuracy,
            "n_evaluated_positive_markers": n_pos_markers,
            "n_evaluated_negative_markers": n_neg_markers,
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


def marker_purity(
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
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute per-cell marker purity using balanced accuracy.

    For each cell of type c:
        - positive markers of c are expected to be expressed.
        - relevant negative markers are negative markers of c that are also
          positive markers of neighboring cell types.

    The score combines:
        - positive_marker_recall: sensitivity, i.e. fraction of positive markers expressed.
        - negative_marker_avoidance: specificity, i.e. fraction of relevant negative markers avoided.
        - marker_balanced_accuracy: mean of sensitivity and specificity.

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
    layer : str | None, optional
        Layer containing count data. If `None`, `adata.X` is used if it looks
        like counts, otherwise `adata.layers["counts"]` is used if available.
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
    inplace : bool, optional, default=True
        If True, store marker purity results in `sdata.tables[tables_key].obs`.

    Returns
    -------
    pandas.DataFrame
        Columns:
            [
             'positive_marker_recall',
             'negative_marker_avoidance',
             'marker_balanced_accuracy',
             'n_evaluated_positive_markers',
             'n_evaluated_negative_markers'
            ]
        indexed by cell ID.
    """
    adata = sdata.tables[tables_key]

    X = _get_count_matrix(adata, layer=layer, tables_key=tables_key)
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    genes = np.asarray(adata.var_names)
    var_index = pd.Index(genes)
    valid_genes = set(var_index)

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
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")

    G = adata.obsp[neighbors_key]
    if sparse.issparse(G):
        G = G.tocsr()
        neighbor_indices = [G[i].indices for i in range(n_cells)]
    else:
        G = np.asarray(G)
        neighbor_indices = [np.where(G[i] > 0)[0] for i in range(n_cells)]

    # Keep only markers that are present in the spatial expression matrix.
    pos_sets = {ct: set(m.get("positive", [])) & valid_genes for ct, m in markers.items()}
    neg_sets = {ct: set(m.get("negative", [])) & valid_genes for ct, m in markers.items()}

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
            "positive_marker_recall": positive_recall,
            "negative_marker_avoidance": negative_avoidance,
            "marker_balanced_accuracy": balanced_accuracy,
            "n_evaluated_positive_markers": n_pos_markers,
            "n_evaluated_negative_markers": n_neg_markers,
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