"""Generate the reference output of `SegTraQ.run_all()` on the ProSeg test dataset.

The ProSeg dataset is used because it is 3D, so the volume metrics (including ovrlpy's
vertical signal integrity and the fraction of heterotypic overlap) are covered as well.

The resulting SpatialData object is written to zarr and is part of the test data. It is
compared against a fresh `run_all()` call in `tests/test_run_all.py`, so after an
intentional change to any metric, re-run this script and upload the new reference.

Usage (from the repository root, after downloading the test data)::

    python tests/generate_run_all_reference.py [--output tests/data/proseg2_run_all.zarr]

This module is also imported by `tests/test_run_all.py`, so that the reference and the
test always use exactly the same inputs and arguments.
"""

import argparse
import warnings
from pathlib import Path

import anndata as ad
from spatialdata import SpatialData

import segtraq as st

DATA_DIR = Path(__file__).parent / "data"
PROSEG_PATH = DATA_DIR / "proseg2.zarr"
ADATA_REF_PATH = DATA_DIR / "scRNAseq_ref_subset.h5ad"
REFERENCE_PATH = DATA_DIR / "proseg2_run_all.zarr"

# same keys as the `sdata_3D` fixture in `tests/conftest.py`
SEGTRAQ_KWARGS = dict(
    points_cell_id_key="assignment",
    points_background_id=4294967295,
    points_gene_key="gene",
    tables_area_key="volume",
    tables_cell_id_key="cell",
    shapes_cell_id_key="cell",
    tables_centroid_x_key="centroid_x",
    tables_centroid_y_key="centroid_y",
    filter_kwargs={"inplace": False},
)

# the reference arguments make run_all perform label transfer once upfront and derive the
# markers for the supervised metrics from the reference, so that every module runs.
RUN_ALL_KWARGS = dict(
    ref_cell_type="celltype",
    ref_raw_counts_layer="raw",
    volume_kwargs={
        "run_ovrlpy": False,
        "heterotypic_overlap_kwargs": {
            "shapes_key_list": [
                "cell_boundaries_z0",
                "cell_boundaries_z1",
                "cell_boundaries_z2",
                "cell_boundaries_z3",
            ]
        },
    },
)


def run_all_on_proseg(
    proseg_path: str | Path = PROSEG_PATH,
    adata_ref_path: str | Path = ADATA_REF_PATH,
) -> SpatialData:
    """Load the ProSeg dataset, run all SegTraQ metrics in place and return the resulting SpatialData."""
    sdata = SpatialData.read(proseg_path)
    adata_ref = ad.read_h5ad(adata_ref_path)

    segtraq_obj = st.SegTraQ(sdata, **SEGTRAQ_KWARGS)

    # run_all only warns when a module fails; the reference has to contain every module
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        segtraq_obj.run_all(adata_ref=adata_ref, inplace=True, **RUN_ALL_KWARGS)

    failures = [
        str(w.message) for w in caught if str(w.message).startswith(("Skipping `run_", "Could not run label transfer"))
    ]
    if failures:
        raise RuntimeError("run_all did not run every module:\n" + "\n".join(failures))

    return segtraq_obj.sdata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proseg", type=Path, default=PROSEG_PATH, help="Path to the ProSeg SpatialData zarr.")
    parser.add_argument("--adata-ref", type=Path, default=ADATA_REF_PATH, help="Path to the scRNA-seq reference.")
    parser.add_argument("--output", type=Path, default=REFERENCE_PATH, help="Where to write the resulting zarr.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output if it already exists.")
    args = parser.parse_args()

    st.settings.n_jobs = -1

    sdata = run_all_on_proseg(args.proseg, args.adata_ref)
    sdata.write(args.output, overwrite=args.overwrite)
    print(f"Wrote run_all reference to {args.output}")


if __name__ == "__main__":
    main()
