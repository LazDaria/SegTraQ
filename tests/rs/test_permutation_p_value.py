import numpy as np
import pytest

from segtraq.rs.utils import _cosine_similarity_rows, _permutation_p_value


def test_permutation_p_value_counts_ulp_ties():
    observed = 0.3
    # values that are mathematically tied with `observed` but differ by a few ULPs
    null = np.array([np.nextafter(observed, 1.0), np.nextafter(observed, -1.0), observed, 0.9, -0.9])
    assert _permutation_p_value(null, observed, tail="lower") == (1 + 4) / 6
    assert _permutation_p_value(null, observed, tail="upper") == (1 + 4) / 6


def test_permutation_p_value_tails():
    null = np.array([0.1, 0.2, 0.3, 0.4])
    assert _permutation_p_value(null, 0.25, tail="lower") == (1 + 2) / 5
    assert _permutation_p_value(null, 0.25, tail="upper") == (1 + 2) / 5
    with pytest.raises(ValueError):
        _permutation_p_value(null, 0.25, tail="both")


def test_reordered_genes_are_ties():
    # Reordering genes (jointly in both profiles) leaves the cosine similarity mathematically
    # unchanged but changes the floating-point summation order. Such null draws must count
    # as ties; with a plain `<=` comparison some of them would not (platform dependent).
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = rng.integers(5, 15)
        p = rng.dirichlet(np.ones(k) * 0.5)
        x_a = rng.multinomial(rng.integers(10, 30), p) + 1
        x_b = rng.multinomial(rng.integers(10, 30), p) + 1
        observed = _cosine_similarity_rows(x_a[None], x_b[None], scale=1e4, n_features_total=300)[0]
        perms = [rng.permutation(k) for _ in range(50)]
        null = _cosine_similarity_rows(
            np.vstack([x_a[o] for o in perms]), np.vstack([x_b[o] for o in perms]), scale=1e4, n_features_total=300
        )
        assert _permutation_p_value(null, observed, tail="lower") == 1.0
