"""Top-level package for SegTraQ."""

__author__ = """Daria Lazic, Matthias Meyer-Bender, Martin Emons"""
__email__ = "daria.lazic@embl.de, matthias.meyerbender@embl.de, martin.emons@uzh.ch"

from . import bl, cs, datasets, pl, ps, rs, sp, vl
from ._settings import settings
from .SegTraQ import SegTraQ
from .utils import (
    bins_to_transcripts,
    cellpose,
    filter_cells,
    markers_from_reference,
    run_label_transfer,
    sample_anomaly_scores,
    validate_spatialdata,
)

# Override canonical module path for Sphinx
for _f in (
    run_label_transfer,
    markers_from_reference,
    validate_spatialdata,
    cellpose,
    filter_cells,
    bins_to_transcripts,
    sample_anomaly_scores,
):
    _f.__module__ = "segtraq"

__all__ = [
    "bl",
    "cs",
    "datasets",
    "rs",
    "sp",
    "pl",
    "vl",
    "ps",
    "markers_from_reference",
    "run_label_transfer",
    "bins_to_transcripts",
    "validate_spatialdata",
    "cellpose",
    "filter_cells",
    "sample_anomaly_scores",
    "SegTraQ",
    "settings",
]
