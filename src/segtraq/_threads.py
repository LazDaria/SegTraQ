import functools
import inspect
import os
from contextlib import contextmanager

import numba
from threadpoolctl import threadpool_limits

from ._settings import settings


@contextmanager
def pinned_threads(n_jobs: int | None = None):
    """Temporarily fix the numba, BLAS and OpenMP thread counts.

    If ``n_jobs`` is None, it will be taken from ``settings.n_jobs``.
    If ``n_jobs`` is -1, maximum parallelism is used.
    """
    if n_jobs is None:
        n_jobs = settings.n_jobs
    if n_jobs == -1:
        n_jobs = len(os.sched_getaffinity(0))
    if n_jobs < 1:
        raise ValueError(
            "n_jobs must be -1 or a positive integer. Set to None to use the default from segtraq.settings.n_jobs."
        )

    old_numba = numba.get_num_threads()
    numba.set_num_threads(min(n_jobs, numba.config.NUMBA_NUM_THREADS))
    try:
        with threadpool_limits(limits=n_jobs):  # OpenBLAS, MKL, OpenMP
            yield
    finally:
        numba.set_num_threads(old_numba)


def with_pinned_threads(func):
    """Run ``func`` inside ``pinned_threads(n_jobs)``, taking ``n_jobs`` from its own signature."""
    sig = inspect.signature(func)
    if "n_jobs" not in sig.parameters:
        raise TypeError(f"{func.__name__} needs an 'n_jobs' parameter to use @with_pinned_threads.")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        with pinned_threads(bound.arguments["n_jobs"]):
            return func(*args, **kwargs)

    return wrapper
