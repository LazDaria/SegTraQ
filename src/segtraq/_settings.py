class Settings:
    def __init__(self):
        self._n_jobs = 1
        self._progress = True

    @property
    def n_jobs(self):
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
