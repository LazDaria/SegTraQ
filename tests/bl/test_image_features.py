import numpy as np
import pandas as pd
import pytest
import spatialdata as sd

import segtraq as st


def test_image_features(sdata_new):
    image_features = st.bl.image_features(sdata_new)
    assert type(image_features) is pd.DataFrame, "Image features should return a DataFrame"
    assert not image_features.empty, "Image features DataFrame should not be empty"


def test_image_features_multiple_channels(sdata_new):
    sdata_multi_channel = sd.deepcopy(sdata_new)
    # add a dummy second channel to the image data
    image = sdata_new.images["image"].values
    # duplicating the first axis of the 3D image array to create a second channel
    sdata_multi_channel.images["image"] = sd.models.Image2DModel.parse(np.repeat(image, 2, axis=0))
    image_features = st.bl.image_features(sdata_multi_channel, channel_names=["DAPI", "dummy_channel"])
    expected_columns = ["DAPI_mean", "DAPI_std", "dummy_channel_mean", "dummy_channel_std"]
    for col in expected_columns:
        assert col in image_features.columns, f"Expected column '{col}' not found in image features DataFrame"


def test_image_features_invalid_key(sdata_new):
    with pytest.raises(ValueError):
        st.bl.image_features(sdata_new, images_key=None)
    with pytest.raises(KeyError):
        st.bl.image_features(sdata_new, images_key="invalid_key")


def test_image_features_incorrect_number_of_channels(sdata_new):
    with pytest.raises(ValueError):
        st.bl.image_features(sdata_new, channel_names=["DAPI", "dummy_channel"])
