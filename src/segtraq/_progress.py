from contextlib import contextmanager

from tqdm.auto import tqdm

from ._settings import settings


class _StepProgress:
    """Progress bar over a fixed number of named steps. No-op if settings.progress is False."""

    def __init__(self, total: int, desc: str, leave: bool = True):
        self.enabled = settings.progress
        self._bar = tqdm(total=total, desc=desc, unit="step", leave=leave, dynamic_ncols=True) if self.enabled else None

    @contextmanager
    def step(self, name: str):
        if self.enabled:
            self._bar.set_postfix_str(f"running: {name}", refresh=True)
        try:
            yield
        finally:
            # also advances if the step raised, so run_all's skip-on-error logic still looks right
            if self.enabled:
                self._bar.update(1)

    def close(self):
        if self.enabled:
            self._bar.set_postfix_str("done", refresh=True)
            self._bar.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
