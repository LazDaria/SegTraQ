from typing import Any

import numpy as np
import ovrlpy
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
    Global z-drift correction (tilt regression).

    This function removes global z-tilt by fitting a plane z ~ x + y and replacing
    z with the residuals.

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


def _run_ovrlpy(
    sdata,
    points_key: str = "transcripts",
    points_gene_key: str = "feature_name",
    points_cell_id_key: str = "cell_id",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_z_key: str = "z",
    n_workers: int = -1,
    random_state: int = 123,
    ovrlpy_init_kwargs: dict[str, Any] | None = None,
    ovrlpy_analyse_kwargs: dict[str, Any] | None = None,
) -> ovrlpy.Ovrlp:
    """

    This helper initializes
    an `ovrlpy.Ovrlp` object, and runs `analyse()`.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object.
    points_key : str, default="transcripts"
        Key in `sdata.points` for the transcript-level points table.
    points_gene_key : str, default="feature_name"
        Column in the points table containing gene names.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript to a cell.
    points_x_key : str, default="x"
        Column in the points table containing transcript x-coordinates.
    points_y_key : str, default="y"
        Column in the points table containing transcript y-coordinates.
    points_z_key : str, default="z"
        Column in the points table containing transcript z-coordinates.
    n_workers : int, default=-1
        Number of workers passed to `ovrlpy.Ovrlp`.
    random_state : int, default=42
        Random seed passed to `ovrlpy.Ovrlp` to ensure reproducible results.
    ovrlpy_init_kwargs : dict or None, default=None
        Additional keyword arguments passed to `ovrlpy.Ovrlp`.
    ovrlpy_analyse_kwargs : dict or None, default=None
        Additional keyword arguments passed to `ovrlpy.Ovrlp.analyse`.

    Returns
    -------
    ovrlpy.Ovrlp
        Initialized and analysed `ovrlpy.Ovrlp` object.
    """

    ovrlpy_init_kwargs = ovrlpy_init_kwargs or {}
    ovrlpy_analyse_kwargs = ovrlpy_analyse_kwargs or {}

    coordinate_df = (
        sdata.points[points_key]
        .rename(
            columns={
                points_gene_key: "gene",
                points_x_key: "x",
                points_y_key: "y",
                points_z_key: "z",
            }
        )
        .loc[:, ["gene", "x", "y", "z", points_cell_id_key]]
        .compute()
    )

    coordinate_df["z"] = coordinate_df["z"] - coordinate_df["z"].min()

    ovrlp_obj = ovrlpy.Ovrlp(
        coordinate_df,
        n_workers=n_workers,
        random_state=random_state,
        **ovrlpy_init_kwargs,
    )

    ovrlp_obj.analyse(**ovrlpy_analyse_kwargs)

    return ovrlp_obj
