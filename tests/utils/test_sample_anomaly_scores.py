import numpy as np
import pandas as pd
import pytest
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest

import segtraq


@pytest.fixture
def metrics():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"transcripts_per_cell": rng.normal(200, 10, 30), "perc_unassigned": rng.normal(10, 1, 30)},
        index=range(1, 31),
    )
    df.loc[7, "perc_unassigned"] = 40  # poor sample: many unassigned transcripts
    df.loc[8, "transcripts_per_cell"] = 400  # unusually good sample
    return df


DIRECTIONS = {"transcripts_per_cell": 1, "perc_unassigned": -1}


def test_poor_sample_ranks_first_and_good_outlier_is_ignored(metrics):
    z, score = segtraq.sample_anomaly_scores(metrics, DIRECTIONS, n_seeds=5, n_trees=100)
    assert score.idxmax() == 7
    assert z.loc[7, "perc_unassigned"] < -3  # direction-corrected: negative = worse
    assert z.loc[8, "transcripts_per_cell"] > 3
    assert score.loc[8] < score.loc[7]
    assert score.rank(ascending=False).loc[8] > 5  # good deviations are clipped


def test_matches_manual_computation(metrics):
    z, score = segtraq.sample_anomaly_scores(metrics, DIRECTIONS, n_seeds=3, n_trees=50)
    X = metrics[list(DIRECTIONS)]
    mad = X.apply(median_abs_deviation, scale="normal")
    expected_z = (X - X.median()) / mad * pd.Series(DIRECTIONS)
    pd.testing.assert_frame_equal(z, expected_z)
    features = expected_z.clip(upper=0)
    expected = -np.mean(
        [IsolationForest(n_estimators=50, random_state=s).fit(features).score_samples(features) for s in range(3)],
        axis=0,
    )
    np.testing.assert_allclose(score.to_numpy(), expected)


def test_rejects_missing_metrics_and_bad_directions(metrics):
    with pytest.raises(KeyError, match="not in the table"):
        segtraq.sample_anomaly_scores(metrics, {"unknown": 1})
    with pytest.raises(ValueError, match="Directions"):
        segtraq.sample_anomaly_scores(metrics, {"perc_unassigned": 2})
