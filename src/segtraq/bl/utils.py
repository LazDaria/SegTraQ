from shapely.geometry import MultiPolygon, Polygon


def count_polygons(geom):
    if isinstance(geom, MultiPolygon):
        return len(geom.geoms)
    elif isinstance(geom, Polygon):
        return 1
    else:
        return 0
