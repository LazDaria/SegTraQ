from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import tifffile as tiff
from rasterio.features import rasterize
from scipy.io import mmread
from scipy.sparse import csr_matrix
from shapely.geometry import mapping
from spatialdata import SpatialData

from .utils import (
    create_spatialdata,
    decompress_geojson,
    labels_mode_projection,
    labels_to_shapes,
    make_adata,
    read_dapi_image,
    read_labels,
    read_shapes,
    read_transcripts,
)


# -----------------------------------------------------------------------------
# Reader: 10x Xenium
# -----------------------------------------------------------------------------
def read_xenium(path_to_data: Path) -> SpatialData:
    """
    Read 10x Xenium data and assemble a SpatialData object.

    Parameters
    ----------
    path_to_data : Path
        Path to the directory containing subset 10x Xenium data.

    Returns
    -------
    SpatialData
        SpatialData object containing transcripts, shapes, labels, tables, and DAPI image.
    """
    # Table
    with gzip.open(path_to_data / "cell_feature_matrix" / "matrix.mtx.gz", "rt") as f:
        X = mmread(f).tocsr()
    features = pd.read_csv(
        path_to_data / "cell_feature_matrix" / "features.tsv.gz", sep="\t", header=None, compression="gzip"
    )
    barcodes = pd.read_csv(
        path_to_data / "cell_feature_matrix" / "barcodes.tsv.gz", sep="\t", header=None, compression="gzip"
    )[0]

    gene_mask = features[2] == "Gene Expression"
    X = X[gene_mask.values, :]
    var = pd.DataFrame(index=features.loc[gene_mask, 1].astype(str).values)
    var.index.name = "gene_symbol"

    meta_df = pd.read_csv(path_to_data / "cells.csv.gz", compression="gzip")
    rename_map = {"cell_centroid_x": "x_centroid", "cell_centroid_y": "y_centroid", "cell_area": "cell_area"}
    for k, v in rename_map.items():
        if k in meta_df.columns and v not in meta_df.columns:
            meta_df[v] = meta_df[k]

    meta_df = meta_df.set_index("cell_id", drop=False)
    common = barcodes[barcodes.isin(meta_df.index)]
    obs = meta_df.loc[common].copy()
    obs.reset_index(drop=True, inplace=True)
    X = X[:, barcodes.isin(meta_df.index).to_numpy()]

    obs["label_id"] = pd.RangeIndex(start=1, stop=len(obs) + 1, dtype=int)
    obs["region"] = pd.Categorical(["cell_labels"] * len(obs))

    adata = make_adata(X.T, obs, var)

    # Image
    dapi = read_dapi_image(path_to_data / "dapi_um.tif")

    # Labels
    labels = read_labels(path_to_data / "cell_mask_um.tif", path_to_data / "nuc_mask_um.tif")

    # Shapes
    shapes = {
        "cell_boundaries": read_shapes(path_to_data / "cell_boundaries.parquet"),
        "nucleus_boundaries": read_shapes(path_to_data / "nucleus_boundaries.parquet"),
    }

    # Points
    transcripts = read_transcripts(
        path_to_data / "transcripts.parquet", rename_map={"x_location": "x", "y_location": "y", "z_location": "z"}
    )

    sdata = create_spatialdata(
        points=transcripts,
        labels=labels,
        shapes=shapes,
        tables=adata,
        images=dapi,
    )
    return sdata


# -----------------------------------------------------------------------------
# Reader: Proseg 2.0
# -----------------------------------------------------------------------------
def read_proseg_2(path_to_proseg_data: Path, path_to_10xdata: Path, consolidate_shapes: bool = True) -> SpatialData:
    """
    Build a SpatialData object from Proseg 2.0 outputs.

    Parameters
    ----------
    path_to_proseg_data : Path
        Path to the directory containing Proseg 2.0 outputs.
    path_to_10xdata : Path
        Path to the directory containing 10x DAPI and nucleus images.
    consolidate_shapes : bool, default=True
        Whether to consolidate shape layers when creating the SpatialData object.

    Returns
    -------
    SpatialData
        SpatialData object containing tables, images, labels, shapes, and transcript points.
    """
    # -------------------------
    # Table (counts + metadata)
    # -------------------------
    counts_df = pd.read_csv(path_to_proseg_data / "expected-counts.csv.gz", compression="gzip")
    var = pd.DataFrame(index=counts_df.columns.astype(str))
    var.index.name = "gene_symbol"

    X_est = csr_matrix(counts_df.values)
    X = csr_matrix(np.rint(counts_df.values).astype(int, copy=False))

    obs = pd.read_csv(path_to_proseg_data / "cell-metadata.csv.gz", compression="gzip")
    obs.drop(columns=["cluster", "scale", "original_cell_id", "population", "fov"], errors="ignore", inplace=True)
    obs.rename(columns={"cell": "cell_id"}, inplace=True)
    obs["cell_id"] = obs["cell_id"] + 1
    obs["label_id"] = obs["cell_id"]
    obs["region"] = pd.Categorical(["cell_labels"] * len(obs))

    adata = make_adata(X, obs, var)
    adata.layers["X_estimated"] = X_est

    # -------------------------
    # Image (DAPI from 10x)
    # -------------------------
    dapi = read_dapi_image(path_to_10xdata / "dapi_um.tif")
    H, W = dapi.shape[1:]

    # -------------------------
    # Polygons → shapes + labels
    # -------------------------
    geojson_path = decompress_geojson(path_to_proseg_data / "cell-polygons-layers.geojson.gz")
    gdf = gpd.read_file(geojson_path)

    gdf["cell"] = gdf["cell"].astype(int)
    gdf["layer"] = gdf["layer"].astype(int)

    labels_dict, shapes_dict = {}, {}
    z_levels = sorted(gdf["layer"].unique())
    stack = np.zeros((len(z_levels), H, W), dtype=np.uint32)

    for zi, z in enumerate(z_levels):
        layer_gdf = gdf[gdf["layer"] == z]

        # Shapes
        layer_shapes = layer_gdf.set_index("cell")["geometry"].to_frame().copy()
        layer_shapes.index.name = "label_id"
        layer_shapes.index += 1
        layer_shapes["cell_id"] = layer_shapes.index
        shapes_dict[f"cell_boundaries_z{int(z)}"] = layer_shapes

        # Labels via rasterize
        shapes_iter = (
            (mapping(geom), int(cid) + 1) for cid, geom in zip(layer_gdf["cell"], layer_gdf.geometry, strict=False)
        )
        img = rasterize(shapes_iter, out_shape=(H, W), fill=0, dtype=np.uint32)
        labels_dict[f"cell_labels_z{int(z)}"] = img
        stack[zi] = img

    # Projection to 2D
    proj = labels_mode_projection(stack)
    labels_dict["cell_labels"] = proj
    shapes_dict["cell_boundaries"] = labels_to_shapes(proj, simplify_tolerance=0.5)

    # Add nucleus labels and shapes
    nucleus_labels = tiff.imread(path_to_10xdata / "nuc_mask_um.tif")
    labels_dict["nucleus_labels"] = nucleus_labels
    nucleus_shapes = read_shapes(path_to_10xdata / "nucleus_boundaries.parquet")
    shapes_dict["nucleus_boundaries"] = nucleus_shapes

    # -------------------------
    # Transcripts
    # -------------------------
    transcripts = read_transcripts(
        path_to_proseg_data / "transcript-metadata.csv.gz",
        rename_map={"gene": "feature_name", "assignment": "cell_id"},
        uint32_max_placeholder=2**32,
    )

    # -------------------------
    # Assemble SpatialData
    # -------------------------
    sdata = create_spatialdata(
        points=transcripts,
        labels=labels_dict,
        shapes=shapes_dict,
        tables=adata,
        images=dapi,
        background_cell_id=0,
        consolidate_shapes=consolidate_shapes,
    )
    return sdata


# -----------------------------------------------------------------------------
# Reader: Proseg 3.0
# -----------------------------------------------------------------------------
def read_proseg_3(path_to_proseg_data: Path, path_to_10xdata: Path, consolidate_shapes: bool = True) -> SpatialData:
    """
    Build a SpatialData object from Proseg 3.0 outputs.

    Parameters
    ----------
    path_to_proseg_data : Path
        Path to the directory containing Proseg 3.0 outputs.
    path_to_10xdata : Path
        Path to the directory containing 10x DAPI and nucleus images.
    consolidate_shapes : bool, default=True
        Whether to consolidate shape layers when creating the SpatialData object.

    Returns
    -------
    SpatialData
        SpatialData object containing tables, images, labels, shapes, and transcript points.
    """
    # -------------------------
    # Table (counts + metadata)
    # -------------------------
    with gzip.open(path_to_proseg_data / "counts.mtx.gz", "rt") as f:
        X = mmread(f).tocsr().astype(np.int32)  # cells x genes

    var_df = pd.read_csv(path_to_proseg_data / "gene-metadata.csv.gz", compression="gzip")
    var = pd.DataFrame(index=var_df["gene"].astype(str).values)
    var.index.name = "gene_symbol"

    obs = pd.read_csv(path_to_proseg_data / "cell-metadata.csv.gz", compression="gzip")
    obs.drop(columns=["cluster", "scale", "original_cell_id"], errors="ignore", inplace=True)
    obs.rename(
        columns={
            "cell": "cell_id",
            "cell_centroid_x": "centroid_x",
            "cell_centroid_y": "centroid_y",
            "cell_area": "cell_area",
        },
        inplace=True,
    )
    obs["cell_id"] = obs["cell_id"] + 1
    obs["label_id"] = obs["cell_id"]
    obs["region"] = pd.Categorical(["cell_labels"] * len(obs))

    adata = make_adata(X, obs, var)

    # -------------------------
    # Image (DAPI from 10x)
    # -------------------------
    dapi = read_dapi_image(path_to_10xdata / "dapi_um.tif")
    H, W = dapi.shape[1:]

    # -------------------------
    # Polygons → shapes + labels
    # -------------------------
    geojson_path = decompress_geojson(path_to_proseg_data / "cell-polygons-layers.geojson.gz")
    gdf = gpd.read_file(geojson_path)

    gdf["cell"] = gdf["cell"].astype(int)
    gdf["layer"] = gdf["layer"].astype(int)

    labels_dict, shapes_dict = {}, {}
    z_levels = sorted(gdf["layer"].unique())
    stack = np.zeros((len(z_levels), H, W), dtype=np.uint32)

    for zi, z in enumerate(z_levels):
        layer_gdf = gdf[gdf["layer"] == z]

        # Shapes
        layer_shapes = layer_gdf.set_index("cell")["geometry"].to_frame().copy()
        layer_shapes.index.name = "label_id"
        layer_shapes.index += 1
        layer_shapes["cell_id"] = layer_shapes.index
        shapes_dict[f"cell_boundaries_z{int(z)}"] = layer_shapes

        # Labels via rasterize
        shapes_iter = (
            (mapping(geom), int(cid) + 1) for cid, geom in zip(layer_gdf["cell"], layer_gdf.geometry, strict=False)
        )
        img = rasterize(shapes_iter, out_shape=(H, W), fill=0, dtype=np.uint32)
        labels_dict[f"cell_labels_z{int(z)}"] = img
        stack[zi] = img

    # Projection to 2D
    proj = labels_mode_projection(stack)
    labels_dict["cell_labels"] = proj
    shapes_dict["cell_boundaries"] = labels_to_shapes(proj, simplify_tolerance=0.5)

    # Add nucleus labels and shapes
    nucleus_labels = tiff.imread(path_to_10xdata / "nuc_mask_um.tif")
    labels_dict["nucleus_labels"] = nucleus_labels
    nucleus_shapes = read_shapes(path_to_10xdata / "nucleus_boundaries.parquet")
    shapes_dict["nucleus_boundaries"] = nucleus_shapes

    # -------------------------
    # Transcripts
    # -------------------------
    transcripts = read_transcripts(
        path_to_proseg_data / "transcript-metadata.csv.gz",
        rename_map={"gene": "feature_name", "assignment": "cell_id"},
        uint32_max_placeholder=2**32,
    )

    # -------------------------
    # Assemble SpatialData
    # -------------------------
    sdata = create_spatialdata(
        points=transcripts,
        labels=labels_dict,
        shapes=shapes_dict,
        tables=adata,
        images=dapi,
        background_cell_id=0,
        consolidate_shapes=consolidate_shapes,
    )
    return sdata


# -----------------------------------------------------------------------------
# Reader: BIDCell
# -----------------------------------------------------------------------------
def read_bidcell(path_to_data: Path, consolidate_shapes: bool = True) -> SpatialData:
    """
    Build a SpatialData object from BIDCell subset data using utility functions.

    Parameters
    ----------
    path_to_data : Path
        Path to the directory containing BIDCell data.
    consolidate_shapes : bool, optional, default=True
        Whether to consolidate shape layers when creating the SpatialData object.

    Returns
    -------
    SpatialData
        SpatialData object containing:
        - `tables`: AnnData with expression matrix, cell metadata, and gene metadata
        - `labels`: cell and nucleus segmentation masks
        - `shapes`: cell and nucleus boundaries derived from labels
        - `images`: DAPI reference image
        - `points`: transcript coordinates and assignments

    Raises
    ------
    FileNotFoundError
        If required CSV or segmentation files are missing.
    ValueError
        If DAPI image has unsupported dimensions.
    """
    # -------------------------
    # Table (expression + metadata)
    # -------------------------
    csv_files = list(path_to_data.glob("cell_gene_matrices/202*/cell*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSVs found under cell_gene_matrices/202*/cell*.csv")

    dfs = [pd.read_csv(f) for f in csv_files]
    merged_df = pd.concat(dfs, ignore_index=True).sort_values("cell_id").reset_index(drop=True)
    merged_df["cell_id"] = merged_df["cell_id"].astype(int)

    # Standardize column names
    merged_df = merged_df.rename(
        columns={
            "cell_size": "cell_area",
            "cell_centroid_x": "centroid_x",
            "cell_centroid_y": "centroid_y",
        }
    )

    meta_cols = ["cell_id", "centroid_x", "centroid_y", "cell_area"]
    expr_cols = [c for c in merged_df.columns if c not in meta_cols]

    obs = merged_df[meta_cols].copy()
    obs["region"] = pd.Categorical(["cell_labels"] * len(obs))
    obs["label_id"] = obs["cell_id"].astype(np.int32)

    var = pd.DataFrame(index=pd.Index(expr_cols, name="gene_symbol"))
    X = merged_df[expr_cols].to_numpy()
    adata = make_adata(X, obs, var)

    # -------------------------
    # Labels (cells + nuclei)
    # -------------------------
    cell_label_files = list(path_to_data.glob("model_outputs/202*/test_output/epoch_4_step_100_connected.tif"))
    if not cell_label_files:
        raise FileNotFoundError("No cell label TIFF found under model_outputs/202*/test_output/")
    cell_labels_path = cell_label_files[0]

    nucleus_labels_path = path_to_data / "nuclei.tif"
    if not nucleus_labels_path.exists():
        raise FileNotFoundError("Missing nucleus labels: nuclei.tif")

    labels = read_labels(cell_labels_path, nucleus_labels_path)
    cell_labels = labels["cell_labels"]
    nucleus_labels = labels["nucleus_labels"]

    # -------------------------
    # Shapes (from labels)
    # -------------------------
    cell_shapes_gdf = labels_to_shapes(cell_labels, simplify_tolerance=0.5)
    nucleus_shapes_gdf = labels_to_shapes(nucleus_labels, simplify_tolerance=0.5)

    # -------------------------
    # Image (DAPI)
    # -------------------------
    dapi_path = path_to_data / "dapi_resized.tif"
    if not dapi_path.exists():
        raise FileNotFoundError("Missing DAPI image: dapi_resized.tif")
    dapi = read_dapi_image(dapi_path)

    # -------------------------
    # Transcripts (points)
    # -------------------------
    transcripts_path = path_to_data / "transcripts_processed.csv"
    if not transcripts_path.exists():
        raise FileNotFoundError("Missing transcripts file: transcripts_processed.csv")

    transcripts_df = read_transcripts(transcripts_path)
    # Map transcripts to cell labels
    x = np.rint(transcripts_df["x"]).astype(int)
    y = np.rint(transcripts_df["y"]).astype(int)
    transcripts_df["cell_id"] = cell_labels[y, x]

    # -------------------------
    # Assemble SpatialData
    # -------------------------
    sdata = create_spatialdata(
        points=transcripts_df,
        labels={"cell_labels": cell_labels, "nucleus_labels": nucleus_labels},
        shapes={"cell_boundaries": cell_shapes_gdf, "nucleus_boundaries": nucleus_shapes_gdf},
        tables=adata,
        images=dapi,
        background_cell_id=0,
        consolidate_shapes=consolidate_shapes,
    )
    return sdata


# -----------------------------------------------------------------------------
# Reader: Segger
# -----------------------------------------------------------------------------
def read_segger(path_to_data: Path, path_to_10xdata: Path, consolidate_shapes: bool = True) -> SpatialData:
    """
    Build a SpatialData object from Segger outputs.

    Parameters
    ----------
    path_to_data : Path
        Path to the directory containing Segger output files.
    path_to_10xdata : Path
        Path to the directory containing 10x DAPI and nucleus images.
    consolidate_shapes : bool, optional, default=True
        Whether to consolidate shape layers when creating the SpatialData object.

    Returns
    -------
    SpatialData
        SpatialData object containing:
        - `tables`: AnnData with expression matrix and cell metadata
        - `labels`: cell and nucleus segmentation masks
        - `shapes`: cell and nucleus boundaries
        - `images`: DAPI reference image
        - `points`: transcript coordinates and assignments
    """

    # -------------------------
    # Table (AnnData)
    # -------------------------
    adata = ad.read_h5ad(path_to_data / "segger_adata.h5ad")
    adata.obs.index.name = "cell_id"
    adata.obs.reset_index(inplace=True)
    adata.obs.sort_values("cell_id", inplace=True)
    adata.obs.reset_index(drop=True, inplace=True)
    adata = adata[adata.obs.index, :].copy()
    adata.obs.drop(columns=["transcripts", "unique_transcripts"], errors="ignore", inplace=True)

    # -------------------------
    # Images
    # -------------------------
    dapi = read_dapi_image(path_to_10xdata / "dapi_um.tif")

    # -------------------------
    # Shapes and labels
    # -------------------------
    # Cell boundaries
    boundaries_gdf = read_shapes(path_to_data / "segger_boundaries.parquet", build_from_vertices=False, backend="gpd")
    boundaries_gdf = boundaries_gdf[boundaries_gdf.geometry.notnull()].copy()

    unique_ids = boundaries_gdf["cell_id"].unique()
    id_str_to_int = {cell_id: i + 1 for i, cell_id in enumerate(unique_ids)}
    boundaries_gdf["label_id"] = boundaries_gdf["cell_id"].map(id_str_to_int)

    cell_shapes_gdf = boundaries_gdf.set_index("label_id")
    cell_shapes_gdf = cell_shapes_gdf[cell_shapes_gdf["cell_id"].isin(adata.obs["cell_id"])]

    # Rasterize cell labels
    H, W = dapi.shape[1:]
    cell_shapes_iter = (
        (mapping(geom), int(cid)) for cid, geom in zip(cell_shapes_gdf.index, cell_shapes_gdf.geometry, strict=False)
    )
    cell_labels = rasterize(cell_shapes_iter, out_shape=(H, W), fill=0, dtype=np.uint32)

    # Nucleus shapes and labels
    nucleus_shapes_gdf = read_shapes(path_to_10xdata / "nucleus_boundaries.parquet")
    nucleus_labels = tiff.imread(path_to_10xdata / "nuc_mask_um.tif")

    # -------------------------
    # Transcripts
    # -------------------------
    transcripts_df = read_transcripts(path_to_data / "segger_transcripts.parquet")
    # Align transcripts with obs
    transcripts_df = transcripts_df[transcripts_df["cell_id"].isin(adata.obs["cell_id"])]

    # -------------------------
    # Finalize table metadata
    # -------------------------
    adata.obs["label_id"] = adata.obs["cell_id"].map(id_str_to_int)
    adata.obs["region"] = pd.Categorical(["cell_labels"] * adata.n_obs)

    # -------------------------
    # Assemble SpatialData
    # -------------------------
    sdata = create_spatialdata(
        points=transcripts_df,
        labels={"cell_labels": cell_labels, "nucleus_labels": nucleus_labels},
        shapes={"cell_boundaries": cell_shapes_gdf, "nucleus_boundaries": nucleus_shapes_gdf},
        tables=adata,
        images=dapi,
        consolidate_shapes=consolidate_shapes,
    )

    return sdata
