import numpy as np
from shapely.geometry import MultiPolygon, Polygon


def count_polygons(geom):
    if isinstance(geom, MultiPolygon):
        return len(geom.geoms)
    elif isinstance(geom, Polygon):
        return 1
    else:
        return 0


def _safe_divide(numerator, denominator):
    """Element-wise division that returns NaN where the denominator is zero."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out
