import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from joblib import Parallel, delayed

from .._settings import settings
from ..utils import _get_genes, _is_background, merge_into_obs, merge_into_uns, merge_into_var
from .utils import count_polygons


def num_cells(sdata: sd.SpatialData, tables_key: str = "table", inplace: bool = True) -> int:
    """
    Counts the number of cells in the given SpatialData object based on the specified table key.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial information and a table.
    tables_key : str, optional
        The key in the `tables` attribute of `sdata` that corresponds to table.
        Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    int
        The number of cells found under the specified table key.
    """
    num_cells = len(sdata.tables[tables_key])
    if inplace:
        merge_into_uns(sdata, tables_key=tables_key, updates={"num_cells": num_cells})
    return num_cells


def num_transcripts(
    sdata: sd.SpatialData, points_key: str = "transcripts", tables_key: str = "table", inplace: bool = True
) -> int:
    """
    Counts the total number of transcripts in the given SpatialData object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing transcript information.
    points_key : str, optional
        The key to access transcript data within the spatial data object. Default is "transcripts".
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    int
        The total number of transcripts in the specified SpatialData object.
    """
    points = sdata.points[points_key]
    points = points.compute() if hasattr(points, "compute") else points
    num_transcripts = len(points)
    if inplace:
        merge_into_uns(sdata, tables_key=tables_key, updates={"num_transcripts": num_transcripts})

    return num_transcripts


def num_genes(
    sdata: sd.SpatialData,
    points_key: str = "transcripts",
    points_gene_key: str = "feature_name",
    tables_key: str = "table",
    inplace: bool = True,
) -> int:
    """
    Counts the number of unique genes in the given SpatialData object.
    This checks both the `points` and `tables` attributes of the SpatialData object to
    ensure that the gene information is consistent across both.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing gene information.
    points_key : str, optional
        The key to access transcript data within the spatial data object. Default is "transcripts".
    points_gene_key : str, optional
        The key to access gene names within the transcript data. Default is "feature_name".
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    int
        The number of unique genes found in the specified SpatialData object.
    """
    # === in the points ===
    # converting from np.int64 to int for consistency
    points = sdata.points[points_key]
    points = points.compute() if hasattr(points, "compute") else points
    num_genes_points = int(points[points_gene_key].nunique())

    # === in the tables ===
    genes_adata = _get_genes(sdata.tables[tables_key])
    num_genes_adata = len(genes_adata)

    # check for consistency between layers
    if num_genes_points != num_genes_adata:
        genes_not_in_points = set(genes_adata) - set(points[points_gene_key].unique())
        genes_not_in_adata = set(points[points_gene_key].unique()) - set(genes_adata)
        genes_not_in_both = list(genes_not_in_points.union(genes_not_in_adata))
        warnings.warn(
            f"The number of genes differs between points ({num_genes_points}) and tables ({num_genes_adata}). "
            f"Example genes that are missed: {genes_not_in_both[: min(5, len(genes_not_in_both))]}. "
            f"If these are control probes, please make sure to include them in the SegTraQ constructor. "
            "For example: SegTraQ(filter_kwargs={'control_prefixes': [...], 'control_genes': [...]}). "
            f"Storing the number of genes from the points layer.",
            stacklevel=2,
        )

    if inplace:
        merge_into_uns(sdata, tables_key=tables_key, updates={"num_genes": num_genes_points})

    return num_genes_points


def perc_unassigned_transcripts(
    sdata: sd.SpatialData,
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: int = -1,
    tables_key: str = "table",
    inplace: bool = True,
) -> float:
    """
    Calculates the percentage of unassigned transcripts in a SpatialData object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The spatial data object containing transcript information.
    points_key : str, optional
        The key to access transcript data within the spatial data object. Default is "transcripts".
    points_cell_id_key : str, optional
        The key to access cell assignment information within the transcript data. Default is "cell_id".
    unassigned_key : int, optional
        The value indicating an unassigned transcript. Default is -1.
    points_background_id : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    float
        The fraction of transcripts that are unassigned.
    """
    points = sdata.points[points_key][points_cell_id_key]
    is_background = _is_background(points, points_background_id)
    is_background = is_background.compute() if hasattr(is_background, "compute") else is_background
    perc_unassigned_transcripts = is_background.mean() * 100

    if inplace:
        merge_into_uns(
            sdata, tables_key=tables_key, updates={"perc_unassigned_transcripts": perc_unassigned_transcripts}
        )

    # converting from np.float to float
    return float(perc_unassigned_transcripts)


def perc_unassigned_transcripts_per_gene(
    sdata: sd.SpatialData,
    points_key: str = "transcripts",
    points_gene_key: str = "feature_name",
    points_cell_id_key: str = "cell_id",
    points_background_id: int = -1,
    tables_key: str = "table",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Calculates the number and percentage of unassigned transcripts per gene in a SpatialData object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The spatial data object containing transcript information.
    points_key : str, optional
        The key to access transcript data within the spatial data object. Default is "transcripts".
    points_gene_key : str, optional
        The key for gene names in the transcript data. Default is "feature_name".
    points_cell_id_key : str, optional
        The key for cell assignment information within the transcript data. Default is "cell_id".
    points_background_id : int, optional
        The value indicating an unassigned transcript. Default is -1.
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, stores the resulting DataFrame in
        `sdata.tables[tables_key].uns["perc_unassigned_transcripts_per_gene"]`.
        Default is True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by gene name with columns:
        - 'total' : total number of transcripts for the gene
        - 'unassigned' : number of unassigned transcripts
        - 'perc_unassigned' : percentage of unassigned transcripts
    """
    points = sdata.points[points_key]

    # Compute only necessary columns
    df = points[[points_gene_key, points_cell_id_key]].compute()

    # Aggregate total and unassigned counts efficiently
    result = (
        df.groupby(points_gene_key, observed=True)[points_cell_id_key]
        .agg(
            total="count",
            unassigned=lambda x: _is_background(x, points_background_id).sum(),
        )
        .astype(int)
    )

    # Compute percentage
    result["perc_unassigned"] = result["unassigned"] / result["total"] * 100

    # Store and return
    if inplace:
        merge_into_var(sdata, tables_key, result)

    return result


def transcripts_per_cell(
    sdata: sd.SpatialData,
    tables_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: int = -1,
    tables_key: str = "table",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Counts the number of transcripts assigned to each cell (excluding unassigned transcripts).

    Parameters
    ----------
    sdata : sd.SpatialData
        A SpatialData object containing transcript and cell assignment information.
    tables_cell_id_key : str
        Column in `sdata.tables[tables_key].obs` containing cell IDs to match with sdata.shapes[shapes_key] index.
    points_key : str, optional
        The key in `sdata.points` corresponding to transcript data. Default is "transcripts".
    points_cell_id_key : str, optional
        The column name in the transcript data that contains cell assignment information. Default is "cell_id".
    points_background_id: int = -1,
        The value indicating an unassigned transcript. Default is -1.
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns: the cell identifier (`cell_key`) and the
        corresponding transcript count ("transcript_count").
    """
    counts = sdata.points[points_key][points_cell_id_key]
    counts = counts.compute() if hasattr(counts, "compute") else counts
    counts = counts[~_is_background(counts, points_background_id)].value_counts().astype("int64")
    counts_df = counts.reset_index()
    counts_df.columns = [points_cell_id_key, "transcript_count"]

    if inplace:
        merge_into_obs(
            sdata, tables_key, counts_df, tables_cell_id_key, points_cell_id_key, fillna_cols=["transcript_count"]
        )

    return counts_df


def genes_per_cell(
    sdata,
    tables_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_gene_key: str = "feature_name",
    points_background_id: int = -1,
    tables_key: str = "table",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Calculates the number of unique genes detected per cell (excluding unassigned transcripts).

    Parameters
    ----------
    sdata : object
        An object containing spatial transcriptomics data with a `points` attribute.
    tables_cell_id_key : str
        Column in `sdata.tables[tables_key].obs` containing cell IDs to match with sdata.shapes[shapes_key] index.
    points_key : str, optional
        The key to access the transcript data within `sdata.points` (default is "transcripts").
    points_cell_id_key : str, optional
        The column name in the transcript data representing cell identifiers (default is "cell_id").
    points_gene_key : str, optional
        The column name in the transcript data representing gene names (default is "feature_name").
    points_background_id: int = -1,
        The value indicating an unassigned transcript. Default is -1.
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per cell, containing the cell identifier and
        the count of unique genes detected in that cell.
    """
    df = sdata.points[points_key].compute()
    # Exclude unassigned transcripts
    df = df[~_is_background(df[points_cell_id_key], points_background_id)]

    # Group by cell and count unique genes
    gene_counts = df.groupby(points_cell_id_key, observed=True)[points_gene_key].nunique().reset_index()
    gene_counts.columns = [points_cell_id_key, "gene_count"]
    if inplace:
        merge_into_obs(
            sdata, tables_key, gene_counts, tables_cell_id_key, points_cell_id_key, fillna_cols=["gene_count"]
        )

    return gene_counts


def mean_transcripts_per_gene_per_cell(
    sdata: sd.SpatialData,
    tables_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_gene_key: str = "feature_name",
    points_background_id: int = -1,
    tables_key: str = "table",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Computes the mean number of transcripts per gene per cell (excluding unassigned transcripts).

    Transcripts are first counted per (cell, gene). Then, for each cell,
    we compute the mean of these per-gene transcript counts across genes
    detected in that cell.

    Notes
    -----
    This mean is computed across *detected* genes only (i.e., genes with at least
    one transcript in the cell). Genes with zero transcripts in a cell are not included.

    Parameters
    ----------
    sdata : object
        An object containing spatial transcriptomics data with a `points` attribute.
    tables_cell_id_key : str
        Column in `sdata.tables[tables_key].obs` containing cell IDs to match with sdata.shapes[shapes_key] index.
    points_key : str, optional
        The key to access the transcript data within `sdata.points` (default is "transcripts").
    points_cell_id_key : str, optional
        The column name in the transcript data representing cell identifiers (default is "cell_id").
    points_gene_key : str, optional
        The column name in the transcript data representing gene names (default is "feature_name").
    points_background_id: int = -1,
        The value indicating an unassigned transcript. Default is -1.
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per cell containing the mean transcripts per detected gene:
        columns are `[points_cell_id_key, "mean_transcripts_per_gene"]`.
    """
    df = sdata.points[points_key].compute()
    # Exclude unassigned transcripts
    df = df[~_is_background(df[points_cell_id_key], points_background_id)]

    # Count transcripts per (cell, gene)
    per_gene_counts = (
        df.groupby([points_cell_id_key, points_gene_key], observed=True).size().reset_index(name="transcript_count")
    )

    # For each cell, compute mean transcripts across detected genes
    per_cell_mean = (
        per_gene_counts.groupby(points_cell_id_key, observed=True)["transcript_count"]
        .mean()
        .reset_index(name="mean_transcripts_per_gene")
    )

    if inplace:
        merge_into_obs(
            sdata,
            tables_key,
            per_cell_mean,
            tables_cell_id_key,
            points_cell_id_key,
            fillna_cols=["mean_transcripts_per_gene"],
        )

    return per_cell_mean


def transcript_density(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_area_key: str = "cell_area",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: int = -1,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Calculates the transcript density for each cell in a SpatialData object.
    Transcript density is defined as the number of transcripts per unit area for each cell.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial transcriptomics data.
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    tables_cell_id_key : str, optional
        The key in the table indicating cell identifiers. Default is "cell_id".
    tables_area_key: str, optional
        The key in the table indicating the cell area. Default is "cell_area".
    points_key : str, optional
        The key to access the transcript data within `sdata.points` (default is "transcripts").
    points_cell_id_key : str, optional
        The column name in the transcript data representing cell identifiers (default is "cell_id").
    points_background_id: int = -1,
        The value indicating an unassigned transcript. Default is -1.
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns `[cell_key, "transcript_density"]`,
        where "transcript_density" is the number of transcripts per unit area for
        each cell. Rows with missing values are dropped.
    """
    # this adds the transcript counts inplace
    # required to compute the density later on
    _ = transcripts_per_cell(
        sdata,
        points_key=points_key,
        tables_cell_id_key=tables_cell_id_key,
        points_background_id=points_background_id,
        points_cell_id_key=points_cell_id_key,
        tables_key=tables_key,
    )

    adata = sdata.tables[tables_key]
    df = adata.obs[[tables_cell_id_key, tables_area_key, "transcript_count"]].copy()
    df["transcript_density"] = df["transcript_count"] / df[tables_area_key]

    if inplace:
        merge_into_obs(
            sdata,
            tables_key,
            df[[tables_cell_id_key, "transcript_density"]],
            tables_cell_id_key,
            tables_cell_id_key,
            fillna_cols=["transcript_density"],
        )

    return df[[tables_cell_id_key, "transcript_density"]].dropna()


def morphological_features(
    sdata: sd.SpatialData,
    tables_cell_id_key: str = "cell_id",
    tables_centroid_x_key: str = "centroid_x",
    tables_centroid_y_key: str = "centroid_y",
    shapes_key: str = "cell_boundaries",
    features_to_compute: list | None = None,
    n_jobs: int | None = None,  # number of parallel jobs, -1 uses all CPUs
    parallel_backend: str = "threading",
    tables_key: str = "table",
    eps: float = 1e-6,
    inplace: bool = True,
):
    """
    Compute morphological features for cell shapes in a spatial transcriptomics dataset.

    Parameters
    ----------
    sdata : object
        Spatial data object containing cell shape information. Must have a `.shapes` attribute with geometries.
    tables_cell_id_key : str
        Column in `sdata.tables[tables_key].obs containing cell IDs to match with `shapes_cell_id_key`.
    tables_centroid_x_key : str, optional
        Column in `sdata.tables[tables_key].obs` to store the x-coordinate of the centroid (default is "centroid_x").
    tables_centroid_y_key : str, optional
        Column in `sdata.tables[tables_key].obs` to store the y-coordinate of the centroid (default is "centroid_y").
    shapes_key : str, optional
        Key in `sdata.shapes` specifying the geometry column (default is "cell_boundaries").
    features_to_compute : list of str, optional
        List of morphological features to compute. If None, all available features are computed.
        Available features: "centroid", "num_polygons", "cell_area", "perimeter", "circularity",
        "solidity", "convexity", "elongation", "eccentricity", "compactness".
    n_jobs : int, optional
        Number of parallel jobs to use for computation.
        Default is None. `-1` uses all available CPU cores.
    parallel_backend : str, optional
        Parallelization backend to use with joblib. Default is "threading".
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    eps : float, optional
        Small constant to avoid division by zero in feature calculations. Default is 1e-6.
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    features : pandas.DataFrame
        DataFrame containing the computed morphological features for each cell, indexed by sdata[shapes_key].index.name.

    Raises
    ------
    ValueError
        If any requested feature in `features_to_compute` is not recognized.

    Notes
    -----
    - Requires `geopandas`, `shapely`, `numpy`, `pandas`, and `joblib`.
    - For multi-part geometries (MultiPolygon), "solidity", "convexity", "elongation", and
      "eccentricity" are computed on the convex hull of the *entire* geometry.
    - Invalid or null geometries are filtered out before computation.
    """
    if n_jobs is None:
        n_jobs = settings.n_jobs

    # Define all possible features
    all_features = [
        "centroid",
        "num_polygons",
        "cell_area",
        "perimeter",
        "circularity",
        "solidity",
        "convexity",
        "elongation",
        "eccentricity",
        "compactness",
    ]
    # if we already have centroids in the table, we do not compute them here
    if features_to_compute is None and (tables_centroid_x_key is not None and tables_centroid_y_key is not None):
        all_features.remove("centroid")

    # If no features specified, compute all
    if features_to_compute is None:
        features_to_compute = all_features
    else:
        # Validate features requested
        invalid_feats = set(features_to_compute) - set(all_features)
        if invalid_feats:
            raise ValueError(f"Unknown features requested: {invalid_feats}")

    cells = sdata.shapes[shapes_key]
    if not isinstance(cells, gpd.GeoDataFrame):
        cells = cells.to_gdf()

    features = pd.DataFrame()

    # in validate_spatialdata(), we ensure that the index of the shapes table is the cell_id
    features[cells.index.name] = cells.index.values

    geom = cells.geometry

    if "centroid" in features_to_compute:
        centroids = geom.centroid
        features["centroid_x"] = centroids.x.values
        features["centroid_y"] = centroids.y.values

    if "num_polygons" in features_to_compute:
        features["num_polygons"] = geom.apply(count_polygons).values

    # Compute features conditionally
    if "cell_area" in features_to_compute or any(
        f in features_to_compute for f in ["circularity", "solidity", "compactness", "sphericity"]
    ):
        areas = geom.area.values
        if "cell_area" in features_to_compute:
            features["cell_area"] = areas
    else:
        areas = None

    if "perimeter" in features_to_compute or any(
        f in features_to_compute
        for f in [
            "circularity",
            "compactness",
            "convexity",
            "sphericity",
        ]
    ):
        perimeters = geom.length.values
        if "perimeter" in features_to_compute:
            features["perimeter"] = perimeters
    else:
        perimeters = None

    if "circularity" in features_to_compute:
        if areas is None:
            areas = geom.area.values
        if perimeters is None:
            perimeters = geom.length.values
        features["circularity"] = 4 * np.pi * areas / (perimeters**2 + eps)

    # convex hull is shared across solidity, convexity, elongation, and eccentricity,
    # so it's computed once here if any of those features are requested
    convex_hull = None
    if any(f in features_to_compute for f in ["solidity", "convexity", "elongation", "eccentricity"]):
        convex_hull = geom.convex_hull

    if "solidity" in features_to_compute or "convexity" in features_to_compute:
        if "solidity" in features_to_compute:
            convex_areas = convex_hull.area.values
            if areas is None:
                areas = geom.area.values
            features["solidity"] = areas / (convex_areas + eps)
        if "convexity" in features_to_compute:
            convex_perimeters = convex_hull.length
            if perimeters is None:
                perimeters = geom.length.values
            features["convexity"] = (convex_perimeters / (perimeters + eps)).values

    # Parallelized elongation and eccentricity calculation, based on the convex hull of the
    # (possibly multi-part) geometry, so fragmented cells aren't reduced to a single sub-polygon
    def compute_elong_ecc(hull):
        if hull.is_empty or hull.area == 0:
            return np.nan, np.nan

        # Compute minimum rotated rectangle
        min_rect = hull.minimum_rotated_rectangle
        coords = list(min_rect.exterior.coords)

        if len(coords) < 4:
            return np.nan, np.nan

        # Compute edge lengths
        edges = [np.linalg.norm(np.array(coords[i]) - np.array(coords[i + 1])) for i in range(4)]
        edges = sorted(edges)
        if edges[1] == 0:
            return np.nan, np.nan

        # Elongation and eccentricity
        elongation = edges[2] / edges[1]
        a = edges[2] / 2
        b = edges[1] / 2
        eccentricity = np.sqrt(a**2 - b**2) / a if a > 0 else np.nan

        return elongation, eccentricity

    if "elongation" in features_to_compute or "eccentricity" in features_to_compute:
        results = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
            delayed(compute_elong_ecc)(hull) for hull in convex_hull
        )
        elongations, eccentricities = zip(*results, strict=False)
        if "elongation" in features_to_compute:
            features["elongation"] = elongations
        if "eccentricity" in features_to_compute:
            features["eccentricity"] = eccentricities

    if "compactness" in features_to_compute:
        if perimeters is None:
            perimeters = geom.length.values
        if areas is None:
            areas = geom.area.values
        features["compactness"] = (perimeters**2) / (areas + eps)

    if inplace:
        merge_into_obs(
            sdata,
            tables_key=tables_key,
            df_to_merge=features,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=cells.index.name,
        )
    return features
