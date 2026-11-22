import numpy as np
from scipy.stats import pearsonr

from segtraq.rs.utils import _cosine_similarity_rows


def test_cosine_similarity_rows_equals_pearson_over_all_genes():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n_genes = int(rng.integers(20, 2000))
        p = rng.dirichlet(np.full(n_genes, 0.05))
        x_a = rng.multinomial(int(rng.integers(20, 2000)), p)
        x_b = rng.multinomial(int(rng.integers(20, 2000)), p)
        active = (x_a + x_b) > 0

        observed = _cosine_similarity_rows(
            x_a[active][None, :],
            x_b[active][None, :],
            scale=1e4,
            n_features_total=n_genes,
        )[0]
        expected = pearsonr(
            np.log1p(1e4 * x_a / x_a.sum()),
            np.log1p(1e4 * x_b / x_b.sum()),
        ).statistic

        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)


def test_similarity_p_value_counts_relabelled_ties():
    # Each gene has at most one transcript, so many permutations only relabel genes
    # and reproduce the observed correlation up to floating-point rounding.
    from segtraq.rs.utils import _two_profile_similarity_metrics

    x_a = np.zeros(500, dtype=int)
    x_b = np.zeros(500, dtype=int)
    x_a[:15] = 1
    x_b[15:30] = 1
    p_value = _two_profile_similarity_metrics(x_a, x_b, rng=np.random.default_rng(0))["similarity_p_value"]
    # Observed profiles share no gene, the most dissimilar arrangement possible, so
    # every permutation is at least as similar and ties must all be counted.
    assert p_value == 1.0
