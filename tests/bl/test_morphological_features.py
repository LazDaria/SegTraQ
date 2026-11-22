import pandas as pd

import segtraq as st

st.settings.n_jobs = -1


def test_morphological_features(sdata_new):
    morphological_features = st.bl.morphological_features(sdata_new)
    assert type(morphological_features) is pd.DataFrame, "Morphological features should return a DataFrame"
    assert not morphological_features.empty, "Morphological features DataFrame should not be empty"
    assert "cell_id" in morphological_features.columns, "DataFrame should contain 'cell_id' column"

    num_cells = st.bl.num_cells(sdata_new)
    assert morphological_features.shape[0] == num_cells, "Number of rows in DataFrame should match number of cells"

    all_features = [
        "cell_area",
        "perimeter",
        "circularity",
        "solidity",
        "convexity",
        "elongation",
        "eccentricity",
        "compactness",
        "num_polygons",
    ]
    # asserts for the different features (that they are present and of correct type, also in the right range)
    for feature in all_features:
        assert feature in morphological_features.columns, f"Feature '{feature}' should be present in DataFrame columns"
        assert feature in sdata_new.tables["table"].obs.columns, (
            f"Feature '{feature}' should be in the observation columns of the table"
        )
        feature_values = morphological_features[feature]
        assert not feature_values.empty, f"Feature '{feature}' should have values"
        assert all(isinstance(value, (int | float)) for value in feature_values), (
            f"Values for '{feature}' should be numeric"
        )
        if feature in ["cell_area", "perimeter", "num_polygons"]:
            assert all(value >= 0 for value in feature_values), f"Values for '{feature}' should be non-negative"
        elif feature in [
            "circularity",
            "solidity",
            "convexity",
            "eccentricity",
        ]:
            assert feature_values.min() >= -1e-6 and feature_values.max() <= 1 + 1e-6, (
                f"Values for '{feature}' should be ~[0, 1]. "
                f"Found min: {feature_values.min()}, max: {feature_values.max()}"
            )


def _toy_sdata(geometries, ids):
    from types import SimpleNamespace

    import geopandas as gpd

    gdf = gpd.GeoDataFrame(geometry=geometries, index=pd.Index(ids, name="cell_id"))
    return SimpleNamespace(shapes={"cell_boundaries": gdf})


RATIO_FEATURES = ["circularity", "solidity", "convexity", "compactness"]


def test_morphological_features_regular_shape_exact():
    import numpy as np
    from shapely.geometry import box

    sdata = _toy_sdata([box(0, 0, 2, 2)], ["square"])
    feats = st.bl.morphological_features(
        sdata, tables_cell_id_key="cell_id", features_to_compute=RATIO_FEATURES, inplace=False
    )
    row = feats.iloc[0]
    assert np.isclose(row["circularity"], np.pi / 4, rtol=0, atol=1e-12)
    assert row["solidity"] == 1.0
    assert row["convexity"] == 1.0
    assert row["compactness"] == 16.0


def test_morphological_features_zero_denominator_is_nan():
    import warnings

    from shapely.geometry import Polygon, box

    # degenerate polygon with zero area (and zero-area convex hull) but non-zero perimeter
    flat = Polygon([(0, 0), (1, 0), (2, 0)])
    sdata = _toy_sdata([flat, box(0, 0, 1, 1)], ["flat", "square"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        feats = st.bl.morphological_features(
            sdata, tables_cell_id_key="cell_id", features_to_compute=RATIO_FEATURES, inplace=False
        ).set_index("cell_id")

    assert pd.isna(feats.loc["flat", "solidity"])
    assert pd.isna(feats.loc["flat", "compactness"])
    # perimeter is non-zero, so circularity and convexity remain defined
    assert feats.loc["flat", "circularity"] == 0.0
    assert not pd.isna(feats.loc["flat", "convexity"])
    assert not feats.loc["square", RATIO_FEATURES].isna().any()


def test_morphological_features_zero_perimeter_is_nan():
    from shapely.geometry import Polygon

    sdata = _toy_sdata([Polygon()], ["empty"])
    feats = st.bl.morphological_features(
        sdata, tables_cell_id_key="cell_id", features_to_compute=RATIO_FEATURES, inplace=False
    )
    assert feats[RATIO_FEATURES].isna().all(axis=None)


def test_morphological_features_no_eps_argument():
    import inspect

    assert "eps" not in inspect.signature(st.bl.morphological_features).parameters
