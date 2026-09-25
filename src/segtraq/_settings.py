class Settings:
    def __init__(self):
        self._n_jobs = 1
        self._progress = True

    @property
    def n_jobs(self):
        # note that for the clustering stability metrics,
        # we use the n_jobs parameter to control the number of threads used by
        # numba, BLAS and OpenMP for PCA, neighbor graph construction and clustering.
        # Results are only reproducible for a fixed n_jobs;
        # the default of 1 gives identical results regardless of the node's CPU allocation.
        # None keeps the libraries' defaults (auto-detected from CPU affinity, not reproducible across machines).
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        if not isinstance(value, int):
            raise TypeError("n_jobs must be an integer.")
        if value == 0 or value < -1:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        self._n_jobs = value

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, value):
        if not isinstance(value, bool):
            raise TypeError("progress must be a bool.")
        self._progress = value


settings = Settings()
