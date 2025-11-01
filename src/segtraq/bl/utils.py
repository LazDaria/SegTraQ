from shapely.geometry import MultiPolygon, Polygon


def count_polygons(geom):
    if isinstance(geom, MultiPolygon):
        return len(geom.geoms)
    elif isinstance(geom, Polygon):
        return 1
    else:
        return 0


def merge_into_obs(sdata, table_key, df_to_merge, on_key, fillna_cols=None):
    obs = sdata.tables[table_key].obs

    # Drop overlapping columns, but keep the merge key
    overlapping = [c for c in df_to_merge.columns if c in obs.columns and c != on_key]
    if overlapping:
        obs = obs.drop(columns=overlapping)

    # Merge
    df = obs.merge(df_to_merge, on=on_key, how="left")

    # Optionally fill numeric columns with zeros
    if fillna_cols:
        for c in fillna_cols:
            if c in df:
                df[c] = df[c].fillna(0)

    sdata.tables[table_key].obs = df
