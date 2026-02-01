"""Top-level package for SegTraQ."""

__author__ = """Daria Lazic, Matthias Meyer-Bender, Martin Emons"""
__email__ = "daria.lazic@embl.de, matthias.meyerbender@embl.de, martin.emons@uzh.ch"

from . import bl, cs, pl, ps, rs, sp, vl
from .SegTraQ import SegTraQ
from .utils import (
    add_nuc_shapes_via_cellpose,
    filter_cells,
    get_ref_markers,
    run_label_transfer,
    validate_spatialdata,
)

# Override canonical module path for Sphinx
for _f in (
    run_label_transfer,
    get_ref_markers,
    validate_spatialdata,
    add_nuc_shapes_via_cellpose,
    filter_cells,
):
    _f.__module__ = "segtraq"

__all__ = [
    "bl",
    "cs",
    "rs",
    "sp",
    "pl",
    "vl",
    "ps",
    "get_ref_markers",
    "get_mut_excl_markers",
    "run_label_transfer",
    "validate_spatialdata",
    "add_nuc_shapes_via_cellpose",
    "filter_cells",
    "SegTraQ",
]
