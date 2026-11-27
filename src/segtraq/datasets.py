"""Example datasets used in the tutorials.

Files are downloaded once and cached. The cache is ``pooch.os_cache("segtraq")`` (for example
``~/.cache/segtraq`` on Linux); set the environment variable ``SEGTRAQ_DATA_DIR`` to use another
folder. Set ``SEGTRAQ_DATA_SOURCE`` to one or more local folders (separated by ``os.pathsep``) that
contain the files to skip the download.

The crops are squares around the fields of view shown in the SegTraQ manuscript:

- human breast, Xenium (GEO GSM9509160): 450 µm around Fig. 4a
- human ovarian cancer, Visium HD (10x Genomics): 500 µm around Fig. 5b
- lung adenocarcinoma, CosMx (GEO GSM9046088): one field of view, or all 36 with SegTraQ metrics
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import anndata as ad
    import spatialdata as sd

# TODO: Zenodo records
TUTORIAL_DATA_URL = "https://zenodo.org/records/TODO/files/"  # SegTraQ tutorial data
ANALYSIS_DATA_URL = "https://zenodo.org/records/TODO/files/"  # segtraq_analysis: SegTraQ-annotated objects

COSMX_ARCHIVE = "segtraq_multisample_cosmx_tma1.zip"

_REGISTRY = {
    "xenium_breast_crop.zarr.zip": "sha256:d9bddc18fdd00da89cfaf3108c059af899a95bd08cfc03be9929bbfe07cf0f51",
    "proseg_breast_crop.zarr.zip": "sha256:013d0378d4657836394cdb02b7393b90c3d33da849aebb5605d581ab284a4be7",
    "proseg_breast_crop_raw.zarr.zip": "sha256:f703d06ef2bf58b8be3c1eca1653d5e340ceca8ecfebb5362aebcabc78e241fb",
    "visium_hd_ovarian_crop.zarr.zip": "sha256:b89a0d243289b5d1044afdad862a62c2bf98f25ca7f62e63f979c8e8cd63de11",
    "visium_hd_ovarian_crop_raw.zarr.zip": "sha256:49aa38693f2080d6cda49d0e98a7cfb8ddf5bbebfdf672531c3bc9bb2cb8951a",
    "cosmx_lung_fov11.zarr.zip": "sha256:4bcf7d2c04b8faf6de1592eb1fc14b6ed2afff272d91bb656daaf704d9edf6d8",
    "reference_breast.h5ad": "sha256:bfb9e331f28e2831044f8e8f90a4d28062561c4203e45ca194eb701a1e0b1c2d",
    "reference_ovarian.h5ad": "sha256:e6c030c74e5d8614bede35d2c6dcce0bb2b489d6716ebaa362d22ba33fcca97b",
    "reference_lung.h5ad": "sha256:9f1930341be89759e12e53a35d159b50ba4631a92c4c9f1ff9fe2ba5c6d9c6e1",
    COSMX_ARCHIVE: "sha256:4b2b3ee8b2e0f4e8c36826d8a69856fd7fb62cf49293576b1c078891654a7625",
}
_URLS = {COSMX_ARCHIVE: ANALYSIS_DATA_URL + COSMX_ARCHIVE}


def cache_dir() -> Path:
    """Folder where downloaded files are stored."""
    import pooch

    return Path(os.environ.get("SEGTRAQ_DATA_DIR", pooch.os_cache("segtraq")))


def fetch(filename: str) -> Path:
    """
    Download a file of the tutorial data (once) and return its local path.

    Zip archives are extracted next to the download; the path of the extracted folder is returned.

    Parameters
    ----------
    filename : str
        Name of the file, one of the keys of ``segtraq.datasets._REGISTRY``.

    Returns
    -------
    Path
        Local path of the file, or of the extracted folder for zip archives.
    """
    if filename not in _REGISTRY:
        raise ValueError(f"Unknown file {filename!r}. Available: {sorted(_REGISTRY)}")

    local = [Path(d) / filename for d in os.environ.get("SEGTRAQ_DATA_SOURCE", "").split(os.pathsep) if d]
    local = [p for p in local if p.exists()]
    if local:
        path = local[0]
    else:
        import pooch

        downloader = pooch.create(
            path=cache_dir(),
            base_url=TUTORIAL_DATA_URL,
            registry=_REGISTRY,
            urls=_URLS,
        )
        path = Path(downloader.fetch(filename, progressbar=True))

    if path.suffix != ".zip":
        return path
    return _unzip(path, cache_dir())


def _unzip(path: Path, target: Path) -> Path:
    """Extract an archive with a single top-level folder into ``target`` (once) and return that folder."""
    with zipfile.ZipFile(path) as zf:
        top = {name.split("/")[0] for name in zf.namelist()}
        if len(top) != 1:
            raise ValueError(f"{path.name} must contain a single top-level folder, found {sorted(top)}")
        folder = target / top.pop()
        done = folder.with_name(folder.name + ".extracted")
        if not done.exists():
            target.mkdir(parents=True, exist_ok=True)
            zf.extractall(target)
            done.touch()
    return folder


def _read_zarr(filename: str) -> sd.SpatialData:
    import spatialdata as sd

    return sd.read_zarr(fetch(filename))


def xenium_breast() -> sd.SpatialData:
    """
    Xenium human breast cancer, Xenium segmentation, 450 µm crop (2,016 cells).

    Contains the DAPI image (``morphology_focus``), transcripts with quality values and z positions,
    cell and nucleus boundaries and the cell x gene table (280 genes). Built from the GEO deposit
    GSM9509160 with the reader of segtraq_analysis (chapter 4), because the deposit lacks files that
    ``spatialdata_io.xenium`` needs.

    Returns
    -------
    SpatialData
    """
    return _read_zarr("xenium_breast_crop.zarr.zip")


def proseg_breast(raw: bool = False) -> sd.SpatialData:
    """
    Proseg 3 segmentation of the same crop as :func:`xenium_breast`.

    Parameters
    ----------
    raw : bool, optional, default=False
        If True, return the object as written by Proseg (coordinates in µm, repositioned transcripts,
        no image or nuclei). If False, return it harmonised with the Xenium data: same coordinate
        system, DAPI image and nucleus boundaries, measured transcript positions and all transcripts.

    Returns
    -------
    SpatialData
    """
    return _read_zarr("proseg_breast_crop_raw.zarr.zip" if raw else "proseg_breast_crop.zarr.zip")


def visium_hd_ovarian(raw: bool = False) -> sd.SpatialData:
    """
    Visium HD human ovarian cancer, Space Ranger cell segmentation, 500 µm crop.

    Parameters
    ----------
    raw : bool, optional, default=False
        If True, return the object as read by ``spatialdata_io.visium_hd`` (2 µm bins and cells, no
        transcripts). If False, return the cells with one transcript per count from
        :func:`segtraq.bins_to_transcripts`.

    Returns
    -------
    SpatialData
    """
    return _read_zarr("visium_hd_ovarian_crop_raw.zarr.zip" if raw else "visium_hd_ovarian_crop.zarr.zip")


def cosmx_lung_fov() -> sd.SpatialData:
    """
    One field of view (FOV 11) of a CosMx lung adenocarcinoma tissue microarray, without SegTraQ metrics.

    Returns
    -------
    SpatialData
    """
    return _read_zarr("cosmx_lung_fov11.zarr.zip")


def cosmx_lung_tma() -> dict[int, sd.SpatialData]:
    """
    All 36 fields of view of the CosMx lung adenocarcinoma tissue microarray, with all SegTraQ metrics.

    This is the archive of the multi-sample analysis in the SegTraQ manuscript (about 670 MB).

    Returns
    -------
    dict of int to SpatialData
        One object per FOV, keyed by FOV number.
    """
    import spatialdata as sd

    folder = fetch(COSMX_ARCHIVE)
    paths = {int(p.stem.rsplit("_", 1)[-1]): p for p in folder.glob("*.zarr")}
    return {fov: sd.read_zarr(paths[fov]) for fov in sorted(paths)}


def reference(tissue: Literal["breast", "ovarian", "lung"]) -> ad.AnnData:
    """
    scRNA-seq reference for supervised metrics, at most 500 cells per cell type, raw counts.

    Parameters
    ----------
    tissue : {"breast", "ovarian", "lung"}
        ``"breast"``: Janesick et al. Chromium, cell types in ``obs["celltype_major"]``.
        ``"ovarian"``: 10x Flex ovarian cancer, cell types in ``obs["transfer_label"]``.
        ``"lung"``: Lee et al. lung adenocarcinoma, cell types in ``obs["author_cell_type_level_1"]``,
        gene symbols in ``var["gene_symbol"]``.

    Returns
    -------
    AnnData
    """
    import anndata as ad

    if tissue not in ("breast", "ovarian", "lung"):
        raise ValueError(f"tissue must be 'breast', 'ovarian' or 'lung', got {tissue!r}")
    return ad.read_h5ad(fetch(f"reference_{tissue}.h5ad"))
