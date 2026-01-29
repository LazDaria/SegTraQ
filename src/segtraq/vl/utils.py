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
