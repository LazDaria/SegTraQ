import numpy as np
import pandas as pd


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
