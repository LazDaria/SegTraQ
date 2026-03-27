import numpy as np
import pandas as pd


def _correct_z_drift(
    tx: pd.DataFrame,
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_z_key: str = "z",
    max_points: int = 1_000_000,
    seed: int | None = 0,
) -> np.ndarray:
    """
    Global z-drift correction (tilt regression + clipping + [0,1] scaling).

    This function removes global z-tilt by fitting a plane z ~ x + y and replacing
    z with the residuals. It then clips extreme residuals to avoid outliers defining
    the depth range and rescales the result to [0,1].

    Parameters
    ----------
    tx : pd.DataFrame
        DataFrame containing transcript coordinates.
    points_x_key, points_y_key, points_z_key : str
        Column names for x/y/z.
    max_points : int, default=1_000_000
        Maximum number of points used to fit the regression (random subsampling).
    seed : int or None, default=0
        Random seed used for subsampling. If None, sampling is not reproducible.

    Returns
    -------
    z_corr : np.ndarray
        Corrected and normalized z values in [0,1], same length/order as `tx`.
    """
    x = tx[points_x_key].to_numpy(dtype=float)
    y = tx[points_y_key].to_numpy(dtype=float)
    z = tx[points_z_key].to_numpy(dtype=float)

    n = len(z)
    n_fit = min(n, int(max_points))

    # fit lstsq only to a subset of the points for comp reasons.
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_fit, replace=False)

    # Fit z ~ x + y + intercept
    A = np.c_[x[idx], y[idx], np.ones(n_fit)]
    coef, _, _, _ = np.linalg.lstsq(A, z[idx], rcond=None)
    wx, wy, b = coef[0], coef[1], coef[2]

    # Residualize full z
    z_resid = z - (b + wx * x + wy * y)

    return z_resid

###### Delete later - since this is already in rs/utils #####
def _norm_log_vector(x: np.ndarray, scale: float = 1e4) -> np.ndarray:
    # Library-size normalize a 1D count vector and apply log1p.
    total = x.sum()
    if total == 0:
        return np.zeros_like(x, dtype=float)
    return np.log1p((x / total) * scale)

def _cosine_sim(x: np.ndarray, y: np.ndarray) -> float:
    """Return cosine similarity between two 1D vectors."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm == 0.0 or y_norm == 0.0:
        return np.nan

    return float(np.dot(x, y) / (x_norm * y_norm))

def _random_partition_counts(
    pooled_counts: np.ndarray,
    n_first: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly split pooled counts into two vectors of sizes `n_first` and
    `pooled_counts.sum() - n_first`, without replacement.
    """
    pooled_counts = np.asarray(pooled_counts)
    pooled_counts = np.rint(pooled_counts).astype(int)

    total = int(pooled_counts.sum())
    if not 0 <= n_first <= total:
        raise ValueError("`n_first` must be between 0 and pooled_counts.sum().")

    if total == 0:
        zeros = np.zeros_like(pooled_counts, dtype=int)
        return zeros, zeros

    # Draw gene counts for the first partition directly, without expanding
    # to transcript-level labels.
    first_counts = rng.multivariate_hypergeometric(pooled_counts, n_first)
    second_counts = pooled_counts - first_counts

    return first_counts, second_counts

################################

def _null_corrected_top_bottom_one_cell(
    x_bottom_raw: np.ndarray,
    x_top_raw: np.ndarray,
    min_transcripts: int = 10,
    min_genes: int = 5,
    n_sim: int = 200,
    scale: float = 1e4,
    random_state: int | None = None,
) -> dict:
    """
    Compute null-corrected cosine similarity between bottom and top for one cell.

    The null is a random partition null:
    pooled bottom+top counts are randomly partitioned into bottom and top
    with the observed totals.

    Parameters
    ----------
    x_bottom_raw : np.ndarray
        Raw bottom gene counts for one cell.
    x_top_raw : np.ndarray
        Raw top gene counts for one cell.
    min_transcripts : int, default=10
        Minimum total transcript count required in both bottom and top.
    min_genes : int, default=5
        Minimum number of genes required after restricting to the shared gene space.
    n_sim : int, default=200
        Number of null simulations.
    scale : float, default=1e4
        Library-size scaling factor applied before log1p.
    random_state : int or None, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary with observed similarity, null summary statistics,
        residual, z-score, and count summaries.
    """
    rng = np.random.default_rng(random_state)

    x_bottom_raw = np.rint(np.asarray(x_bottom_raw)).astype(int)
    x_top_raw = np.rint(np.asarray(x_top_raw)).astype(int)

    # Keep genes present in at least one part.
    mask = (x_bottom_raw + x_top_raw) > 0
    x_bottom_raw = x_bottom_raw[mask]
    x_top_raw = x_top_raw[mask]

    n_genes_used = int(mask.sum())
    n_bottom = int(x_bottom_raw.sum())
    n_top = int(x_top_raw.sum())

    result = {
        "similarity_top_bottom": np.nan,
        "similarity_top_bottom_null_mean": np.nan,
        "similarity_top_bottom_null_sd": np.nan,
        "similarity_top_bottom_residual": np.nan,
        "similarity_top_bottom_zscore": np.nan,
        "bottom_counts_used": n_bottom,
        "top_counts_used": n_top,
        "n_genes_used": n_genes_used,
    }

    if n_genes_used < min_genes or n_bottom < min_transcripts or n_top < min_transcripts:
        return result

    x_bottom = _norm_log_vector(x_bottom_raw, scale=scale)
    x_top = _norm_log_vector(x_top_raw, scale=scale)

    sim_obs = _cosine_sim(x_bottom, x_top)

    pooled = x_bottom_raw + x_top_raw
    sims_null = np.empty(n_sim, dtype=float)

    for i in range(n_sim):
        sim_bottom_raw, sim_top_raw = _random_partition_counts(
            pooled_counts=pooled,
            n_first=n_bottom,
            rng=rng,
        )

        sim_bottom = _norm_log_vector(sim_bottom_raw, scale=scale)
        sim_top = _norm_log_vector(sim_top_raw, scale=scale)

        sims_null[i] = _cosine_sim(sim_bottom, sim_top)

    null_mean = float(np.mean(sims_null))
    null_sd = float(np.std(sims_null, ddof=1)) if n_sim > 1 else 0.0
    residual = float(sim_obs - null_mean)
    zscore = np.nan if np.isclose(null_sd, 0.0) else float(residual / null_sd)

    result.update(
        {
            "similarity_top_bottom": float(sim_obs),
            "similarity_top_bottom_null_mean": null_mean,
            "similarity_top_bottom_null_sd": null_sd,
            "similarity_top_bottom_residual": residual,
            "similarity_top_bottom_zscore": zscore,
        }
    )

    return result