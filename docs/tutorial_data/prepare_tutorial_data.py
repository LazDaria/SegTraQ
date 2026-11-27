"""Build the tutorial data from the outputs of the segtraq_analysis pipelines.

The inputs are only read. Everything is written to ``--out``.

    python prepare_tutorial_data.py --analysis-data DATA --out OUT xenium proseg visium cosmx references
    python prepare_tutorial_data.py --analysis-data DATA --out OUT visium_raw   # needs ~256 GB of memory
    python prepare_tutorial_data.py --analysis-data DATA --out OUT package      # files to upload to Zenodo

``DATA`` is the ``data/`` folder of https://github.com/LazDaria/segtraq_analysis after running the
pipelines of chapters 3, 4 and 5. ``visium_raw`` reads the full Space Ranger outputs with
spatialdata-io; run it as a cluster job.

Crops are squares around the centre of the field of view shown in the manuscript (Fig. 4a and 5b),
made larger so that they are representative of the sample but still quick to analyse.
"""

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import spatialdata as sd

# write strings in the format that all anndata versions read
ad.settings.allow_write_nullable_strings = False

# Xenium human breast (chapter 4). Fig. 4a shows a 150 µm square with top-left corner (2330.125, 1755.0625) µm.
XENIUM_PIXEL_SIZE = 0.2125  # µm per pixel of the morphology image
BREAST_CENTRE_UM = (2330.125 + 75, 1755.0625 + 75)
BREAST_SIZE_UM = 450  # 3x the side of Fig. 4a

# Visium HD human ovarian cancer (chapter 5). Fig. 5b shows a 250 µm square with top-left corner
# (10895.0, 12664.9) in full-resolution pixels.
VISIUM_DATASET = "Visium_HD_Human_Ovarian_Cancer_FF"
VISIUM_MPP = 0.2737973928159521  # µm per full-resolution pixel (scalefactors_json.json)
VISIUM_CENTRE_PX = (10895.0 + 125 / VISIUM_MPP, 12664.9 + 125 / VISIUM_MPP)
VISIUM_SIZE_UM = 500  # 2x the side of Fig. 5b

# CosMx lung adenocarcinoma (chapter 3): one FOV without SegTraQ metrics.
COSMX_FOV = 11
COSMX_PREFIX = "GSM9046088_Lung_Adenocarcinoma_TMA1_CosMx"

# scRNA-seq references: at most this many cells per cell type (all genes are kept).
MAX_CELLS_PER_TYPE = 500
SEED = 0


def crop(sdata, centre, size, coordinate_system):
    """Square bounding-box query. Points inside the box are kept; cells are kept if their shape
    intersects the box, and the table keeps the rows of these cells."""
    half = size / 2
    return sdata.query.bounding_box(
        axes=("x", "y"),
        min_coordinate=[centre[0] - half, centre[1] - half],
        max_coordinate=[centre[0] + half, centre[1] + half],
        target_coordinate_system=coordinate_system,
        filter_table=True,
    )


def write(sdata, path):
    # a crop keeps the partitions of the full dataset, most of them empty: merge them into one
    for key, points in sdata.points.items():
        merged = points.repartition(npartitions=1)
        merged.attrs = points.attrs
        sdata.points[key] = merged
    if path.exists():
        shutil.rmtree(path)
    sdata.write(path)
    print(f"wrote {path}")
    print(sdata)


def breast_box_px():
    centre = tuple(c / XENIUM_PIXEL_SIZE for c in BREAST_CENTRE_UM)
    return centre, BREAST_SIZE_UM / XENIUM_PIXEL_SIZE


def xenium(data, out):
    """Xenium segmentation, built from the GEO deposit by segtraq_analysis (03_xenium_to_spatialdata.py),
    because the deposit lacks the cells.zarr.zip that spatialdata_io.xenium needs."""
    src = (
        data
        / "04_segmentation_method_comparison/segmentation_results/xenium/xenium_breast_breast_B2_B2_xenium_seg.zarr"
    )
    centre, size = breast_box_px()
    write(crop(sd.read_zarr(src), centre, size, "global"), out / "xenium_breast_crop.zarr")


def proseg(data, out):
    """Proseg 3 on the same Xenium sample: the raw Proseg output (coordinates in µm) and the
    harmonised object (Xenium image, nuclei, measured coordinates and all transcripts)."""
    base = data / "04_segmentation_method_comparison/segmentation_results/proseg"
    raw = sd.read_zarr(base / "xenium_breast_breast_B2_B2_proseg3.zarr")
    write(crop(raw, BREAST_CENTRE_UM, BREAST_SIZE_UM, "global"), out / "proseg_breast_crop_raw.zarr")

    harmonised = sd.read_zarr(base / "xenium_breast_breast_B2_B2_proseg.zarr")
    centre, size = breast_box_px()
    write(crop(harmonised, centre, size, "global"), out / "proseg_breast_crop.zarr")


def visium(data, out):
    """Visium HD cells with transcripts from bins_to_transcripts, as used for SegTraQ in chapter 5."""
    src = data / f"05_sequencing_based/{VISIUM_DATASET}/sdata/visium_eval.zarr"
    sdata = crop(sd.read_zarr(src), VISIUM_CENTRE_PX, VISIUM_SIZE_UM / VISIUM_MPP, VISIUM_DATASET)
    write(sdata, out / "visium_hd_ovarian_crop.zarr")


def visium_raw(data, out):
    """The default spatialdata-io object (2 µm bins and Space Ranger cells), before bins_to_transcripts."""
    import spatialdata_io

    sdata = spatialdata_io.visium_hd(
        data / f"05_sequencing_based/{VISIUM_DATASET}",
        load_nucleus_segmentations=True,
        filtered_counts_file=False,  # as in chapter 5: all bins, to match Proseg
        bin_size=2,
    )
    sdata = crop(sdata, VISIUM_CENTRE_PX, VISIUM_SIZE_UM / VISIUM_MPP, VISIUM_DATASET)
    write(sdata, out / "visium_hd_ovarian_crop_raw.zarr")


def cosmx(data, out):
    """One CosMx FOV without SegTraQ metrics. The 36 FOVs with metrics are the chapter 3 archive."""
    src = data / f"03_multi_sample_analysis/fov_zarrs/{COSMX_PREFIX}_{COSMX_FOV}.zarr"
    write(sd.read_zarr(src), out / f"cosmx_lung_fov{COSMX_FOV}.zarr")


def subsample(adata, cell_type_key, keep_obs):
    rng = np.random.default_rng(SEED)
    keep = []
    for idx in adata.obs.groupby(cell_type_key, observed=True).indices.values():
        keep.extend(rng.choice(idx, size=min(len(idx), MAX_CELLS_PER_TYPE), replace=False))
    sub = adata[np.sort(keep)].to_memory() if adata.isbacked else adata[np.sort(keep)].copy()
    sub.obs = sub.obs[keep_obs].copy()
    sub.obsm.clear()
    sub.uns.clear()
    return sub


def references(data, out):
    refs = {
        "reference_breast.h5ad": (
            "04_segmentation_method_comparison/data/reference/prepared/Janesick_breastCancer_Chromium.h5ad",
            "celltype_major",
            ["Annotation", "celltype_major"],
        ),
        "reference_ovarian.h5ad": (
            "05_sequencing_based/17k_Ovarian_Cancer_scFFPE/reference.h5ad",
            "transfer_label",
            ["Cell Annotation", "transfer_label"],
        ),
        "reference_lung.h5ad": (
            "03_multi_sample_analysis/rna_seq/lung_adenocarcinoma_scrnaseq.h5ad",
            "author_cell_type_level_1",
            ["author_cell_type_level_1", "author_cell_type_level_2"],
        ),
    }
    for name, (src, key, keep_obs) in refs.items():
        sub = subsample(ad.read_h5ad(data / src, backed="r"), key, keep_obs)
        sub.write_h5ad(out / name, compression="gzip")
        print(f"wrote {out / name}: {sub.n_obs} cells x {sub.n_vars} genes")
        print(sub.obs[key].value_counts().to_string())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def package(data, out):
    """One zip per SpatialData object (top-level folder <name>.zarr) and the references as they are,
    in out/zenodo, with their SHA256 hashes for the registry in segtraq.datasets."""
    upload = out / "zenodo"
    upload.mkdir(exist_ok=True)
    files = []
    for zarr in sorted(out.glob("*.zarr")):
        target = upload / f"{zarr.name}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(zarr.rglob("*")):
                zf.write(f, f.relative_to(out))
        files.append(target)
    for h5ad in sorted(out.glob("*.h5ad")):
        shutil.copy2(h5ad, upload / h5ad.name)
        files.append(upload / h5ad.name)
    lines = [f"{f.name} sha256:{sha256(f)}" for f in files]
    (upload / "registry.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


STEPS = {f.__name__: f for f in (xenium, proseg, visium, visium_raw, cosmx, references, package)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-data", type=Path, required=True, help="data/ folder of segtraq_analysis")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("steps", nargs="+", choices=list(STEPS))
    args = parser.parse_args()
    chapters = [p.resolve() for p in args.analysis_data.iterdir() if p.name[:2].isdigit()]
    if any(args.out.resolve().is_relative_to(c) for c in chapters):
        raise SystemExit("--out must not be inside a chapter folder of --analysis-data")
    args.out.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        STEPS[step](args.analysis_data, args.out)
