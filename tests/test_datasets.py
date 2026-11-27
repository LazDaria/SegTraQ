import zipfile

import pytest

from segtraq import datasets


def _make_zip(path, top="example.zarr"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{top}/zarr.json", "{}")
        zf.writestr(f"{top}/tables/zarr.json", "{}")


def test_fetch_uses_local_source_and_extracts_once(tmp_path, monkeypatch):
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    _make_zip(source / "xenium_breast_crop.zarr.zip", top="xenium_breast_crop.zarr")
    monkeypatch.setenv("SEGTRAQ_DATA_SOURCE", str(source))
    monkeypatch.setenv("SEGTRAQ_DATA_DIR", str(cache))

    folder = datasets.fetch("xenium_breast_crop.zarr.zip")
    assert folder == cache / "xenium_breast_crop.zarr"
    assert (folder / "tables" / "zarr.json").exists()

    # a second call does not extract again
    (folder / "tables" / "zarr.json").unlink()
    assert datasets.fetch("xenium_breast_crop.zarr.zip") == folder
    assert not (folder / "tables" / "zarr.json").exists()


def test_fetch_returns_non_zip_files_as_they_are(tmp_path, monkeypatch):
    (tmp_path / "reference_lung.h5ad").write_bytes(b"")
    monkeypatch.setenv("SEGTRAQ_DATA_SOURCE", str(tmp_path))
    assert datasets.fetch("reference_lung.h5ad") == tmp_path / "reference_lung.h5ad"


def test_fetch_rejects_unknown_files():
    with pytest.raises(ValueError, match="Unknown file"):
        datasets.fetch("unknown.zip")


def test_unzip_requires_single_top_level_folder(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a/x", "")
        zf.writestr("b/y", "")
    with pytest.raises(ValueError, match="single top-level folder"):
        datasets._unzip(path, tmp_path / "out")


def test_reference_rejects_unknown_tissue():
    with pytest.raises(ValueError, match="tissue"):
        datasets.reference("liver")
