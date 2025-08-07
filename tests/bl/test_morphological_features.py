import pandas as pd

import segtraq as st


def test_morphological_features(sdata_new):
    morphological_features = st.bl.morphological_features(sdata_new)
    assert type(morphological_features) is pd.DataFrame, "Morphological features should return a DataFrame"
    assert not morphological_features.empty, "Morphological features DataFrame should not be empty"
    assert "cell_id" in morphological_features.columns, "DataFrame should contain 'cell_id' column"

    # TODO: reactivate once the test data is updated
    # num_cells = st.bl.num_cells(sdata_new)
    # assert morphological_features.shape[0] == num_cells, "Number of rows in DataFrame should match number of cells"

    all_features = [
        "cell_area",
        "perimeter",
        "circularity",
        "bbox_width",
        "bbox_height",
        "extent",
        "solidity",
        "convexity",
        "elongation",
        "eccentricity",
        "compactness",
        "sphericity",
    ]
    # asserts for the different features (that they are present and of correct type, also in the right range)
    for feature in all_features:
        assert feature in morphological_features.columns, f"Feature '{feature}' should be present in DataFrame columns"
        feature_values = morphological_features[feature]
        assert not feature_values.empty, f"Feature '{feature}' should have values"
        assert all(isinstance(value, (int | float)) for value in feature_values), (
            f"Values for '{feature}' should be numeric"
        )
        if feature in ["cell_area", "perimeter", "bbox_width", "bbox_height"]:
            assert all(value >= 0 for value in feature_values), f"Values for '{feature}' should be non-negative"
        elif feature in [
            "circularity",
            "extent",
            "solidity",
            "convexity",
            "eccentricity",
            "sphericity",
        ]:
            assert feature_values.min() >= -1e-6 and feature_values.max() <= 1 + 1e-6, (
                f"Values for '{feature}' should be ~[0, 1]. "
                f"Found min: {feature_values.min()}, max: {feature_values.max()}"
            )
