import functools
import inspect
from contextlib import contextmanager

import numba
from threadpoolctl import threadpool_limits


@contextmanager
def pinned_threads(n_threads: int | None = 1):
    """Temporarily fix the numba, BLAS and OpenMP thread counts.

    If ``n_threads`` is None, nothing is changed (libraries auto-detect from CPU affinity).
    """
    if n_threads is None:
        yield
        return
    if n_threads < 1:
        raise ValueError(f"n_threads must be >= 1 or None, got {n_threads}.")

    old_numba = numba.get_num_threads()
    numba.set_num_threads(min(n_threads, numba.config.NUMBA_NUM_THREADS))
    try:
        with threadpool_limits(limits=n_threads):  # OpenBLAS, MKL, OpenMP
            yield
    finally:
        numba.set_num_threads(old_numba)


def with_pinned_threads(func):
    """Run ``func`` inside ``pinned_threads(n_threads)``, taking ``n_threads`` from its own signature."""
    sig = inspect.signature(func)
    if "n_threads" not in sig.parameters:
        raise TypeError(f"{func.__name__} needs an 'n_threads' parameter to use @with_pinned_threads.")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        with pinned_threads(bound.arguments["n_threads"]):
            return func(*args, **kwargs)

    return wrapper
