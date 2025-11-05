import math

import pandas as pd


def _is_missing(x):
    """Return True for any kind of NA / NaN / None."""
    try:
        # Works for np.nan, float('nan'), pd.NA, pd.NaT, None
        return pd.isna(x) or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False
