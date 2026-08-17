"""Top-level package for SegTraQ."""

__author__ = """Daria Lazic, Matthias Meyer-Bender, Martin Emons"""
__email__ = "daria.lazic@embl.de, matthias.meyerbender@embl.de, martin.emons@uzh.ch"

from . import bl, cs, pl, ps, rd, sp, vl
from .SegTraQ import SegTraQ
from .utils import (
    bins_to_transcripts,
    cellpose,
    filter_cells,
    markers_from_reference,
    run_label_transfer,
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
):
    _f.__module__ = "segtraq"

__all__ = [
    "bl",
    "cs",
    "rd",
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
    "SegTraQ",
]
