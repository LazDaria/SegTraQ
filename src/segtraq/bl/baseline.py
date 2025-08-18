import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from joblib import Parallel, delayed
from shapely.geometry import Polygon


def num_cells(sdata: sd.SpatialData, shape_key: str = "cell_boundaries") -> int:
    """
    Counts the number of cells in the given SpatialData object based on the specified shape key.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial information and cell boundaries.
    shape_key : str, optional
        The key in the `shapes` attribute of `sdata` that corresponds to cell boundaries.
        Default is "cell_boundaries".

    Returns
    -------
    int
        The number of cells found under the specified shape key.
    """
    return len(sdata.shapes[shape_key])


def num_transcripts(sdata: sd.SpatialData, transcript_key: str = "transcripts"):
    """
    Counts the total number of transcripts in the given SpatialData object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing transcript information.
    transcript_key : str, optional
        The key to access transcript data within the spatial data object. Default is "transcripts".

    Returns
    -------
    int
        The total number of transcripts in the specified SpatialData object.
    """
    return sdata.points[transcript_key].shape[0].compute()


def num_genes(
    sdata: sd.SpatialData,
    transcript_key: str = "transcripts",
    gene_key: str = "feature_name",
) -> int:
    """
    Counts the number of unique genes in the given SpatialData object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing gene information.
    transcript_key : str, optional
        The key to access transcript data within the spatial data object. Default is "transcripts".
    gene_key : str, optional
        The key to access gene names within the transcript data. Default is "feature_name".

    Returns
    -------
    int
        The number of unique genes found in the specified SpatialData object.
    """
    # converting from np.int64 to int for consistency
    return int(sdata.points[transcript_key][gene_key].nunique().compute())


def perc_unassigned_transcripts(
    sdata: sd.SpatialData,
    transcript_key: str = "transcripts",
    cell_key: str = "cell_id",
    unassigned_key: int = -1,
) -> float:
    """
    Calculates the proportion of unassigned transcripts in a SpatialData object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The spatial data object containing transcript information.
    transcript_key : str, optional
        The key to access transcript data within the spatial data object. Default is "transcripts".
    cell_key : str, optional
        The key to access cell assignment information within the transcript data. Default is "cell_id".
    unassigned_key : int, optional
        The value indicating an unassigned transcript. Default is -1.

    Returns
    -------
    float
        The fraction of transcripts that are unassigned.
    """
    counts = sdata.points[transcript_key][cell_key].compute().value_counts()
    num_unassigned = counts.get(unassigned_key, 0)
    # converting from np.float64 to float for consistency
    return float(num_unassigned / counts.sum())


def transcripts_per_cell(
    sdata: sd.SpatialData,
    transcript_key: str = "transcripts",
    cell_key: str = "cell_id",
) -> pd.DataFrame:
    """
    Counts the number of transcripts assigned to each cell.

    Parameters
    ----------
    sdata : sd.SpatialData
        A SpatialData object containing transcript and cell assignment information.
    transcript_key : str, optional
        The key in `sdata.points` corresponding to transcript data. Default is "transcripts".
    cell_key : str, optional
        The column name in the transcript data that contains cell assignment information. Default is "cell_id".

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns: the cell identifier (`cell_key`) and the
        corresponding transcript count ("transcript_count").
    """
    counts = sdata.points[transcript_key][cell_key].compute().value_counts()
    counts_df = counts.reset_index()
    counts_df.columns = [cell_key, "transcript_count"]
    return counts_df


def genes_per_cell(sdata, transcript_key="transcripts", cell_key="cell_id", gene_key="feature_name"):
    """
    Calculates the number of unique genes detected per cell.

    Parameters
    ----------
    sdata : object
        An object containing spatial transcriptomics data with a `points` attribute.
    transcript_key : str, optional
        The key to access the transcript data within `sdata.points` (default is "transcripts").
    cell_key : str, optional
        The column name in the transcript data representing cell identifiers (default is "cell_id").
    gene_key : str, optional
        The column name in the transcript data representing gene names (default is "feature_name").

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per cell, containing the cell identifier and
        the count of unique genes detected in that cell.
    """
    df = sdata.points[transcript_key].compute()
    # Group by cell and count unique genes
    gene_counts = df.groupby(cell_key)[gene_key].nunique().reset_index()
    gene_counts.columns = [cell_key, "gene_count"]
    return gene_counts


def transcript_density(
    sdata: sd.SpatialData,
    table_key: str = "table",
    transcript_key: str = "transcripts",
    cell_key: str = "cell_id",
) -> pd.DataFrame:
    """
    Calculates the transcript density for each cell in a SpatialData object.
    Transcript density is defined as the number of transcripts per unit area for each cell.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial transcriptomics data.
    table_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    transcript_key : str, optional
        The key in the transcript table indicating transcript identifiers. Default is "transcripts".
    cell_key : str, optional
        The key in the table indicating cell identifiers. Default is "cell_id".

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns `[cell_key, "transcript_density"]`,
        where "transcript_density" is the number of transcripts per unit area for
        each cell. Rows with missing values are dropped.

    Notes
    -----
    Requires that the input AnnData table contains a "cell_area" column in `.obs`.
    """
    adata = sdata.tables[table_key]
    counts_df = transcripts_per_cell(sdata, transcript_key, cell_key)
    area_df = adata.obs[[cell_key, "cell_area"]]

    merged = counts_df.merge(area_df, on=cell_key, how="left")
    merged["transcript_density"] = merged["transcript_count"] / merged["cell_area"]

    return merged[[cell_key, "transcript_density"]].dropna()


def morphological_features(
    sdata,
    shape_key: str = "cell_boundaries",
    id_key: str = "cell_id",
    features_to_compute: list = None,
    n_jobs: int = -1,  # number of parallel jobs, -1 uses all CPUs
):
    """
    Compute morphological features for cell shapes in a spatial transcriptomics dataset.

    Parameters
    ----------
    sdata : object
        Spatial data object containing cell shape information. Must have a `.shapes` attribute with geometries.
    shape_key : str, optional
        Key in `sdata.shapes` specifying the geometry column (default is "cell_boundaries").
    id_key : str, optional
        Key in `sdata.shapes` specifying the unique cell identifier column (default is "cell_id").
    features_to_compute : list of str, optional
        List of morphological features to compute. If None, all available features are computed.
        Available features: "cell_area", "perimeter", "circularity", "bbox_width", "bbox_height",
        "extent", "solidity", "convexity", "elongation", "eccentricity", "compactness", "sphericity".
    n_jobs : int, optional
        Number of parallel jobs to use for computation. -1 uses all available CPUs (default is -1).

    Returns
    -------
    features : pandas.DataFrame
        DataFrame containing the computed morphological features for each cell, indexed by `id_key`.

    Raises
    ------
    ValueError
        If any requested feature in `features_to_compute` is not recognized.

    Notes
    -----
    - Requires `geopandas`, `shapely`, `numpy`, `pandas`, and `joblib`.
    - Some features are proxies or approximations (e.g., "sphericity" uses "circularity").
    - Invalid or null geometries are filtered out before computation.
    """
    # Define all possible features
    all_features = [
        "cell_area",
        "perimeter",
        "circularity",
        "bbox_width",
        "bbox_height",
        "extent",
        "solidity",
        "convexity",
        "elongation",
        "eccentricity",
        "compactness",
    ]

    # If no features specified, compute all
    if features_to_compute is None:
        features_to_compute = all_features
    else:
        # Validate features requested
        invalid_feats = set(features_to_compute) - set(all_features)
        if invalid_feats:
            raise ValueError(f"Unknown features requested: {invalid_feats}")

    cells = sdata.shapes[shape_key]
    if not isinstance(cells, gpd.GeoDataFrame):
        cells = cells.to_gdf()

    # Filter valid geometries
    cells = cells[cells.geometry.notnull() & cells.geometry.is_valid].copy().reset_index()

    features = pd.DataFrame()
    features[id_key] = cells[id_key].values
    geom = cells.geometry

    # Compute features conditionally
    if "cell_area" in features_to_compute or any(
        f in features_to_compute for f in ["circularity", "extent", "solidity", "compactness", "sphericity"]
    ):
        areas = geom.area
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
            "compactness",
            "sphericity",
        ]
    ):
        perimeters = geom.length
        if "perimeter" in features_to_compute:
            features["perimeter"] = perimeters
    else:
        perimeters = None

    if "circularity" in features_to_compute:
        if areas is None:
            areas = geom.area
        if perimeters is None:
            perimeters = geom.length
        features["circularity"] = 4 * np.pi * areas / (perimeters**2 + 1e-6)

    if any(f in features_to_compute for f in ["bbox_width", "bbox_height", "extent"]):
        bounds = geom.bounds
        if "bbox_width" in features_to_compute:
            features["bbox_width"] = bounds["maxx"] - bounds["minx"]
        if "bbox_height" in features_to_compute:
            features["bbox_height"] = bounds["maxy"] - bounds["miny"]
        if "extent" in features_to_compute:
            width = bounds["maxx"] - bounds["minx"]
            height = bounds["maxy"] - bounds["miny"]
            if areas is None:
                areas = geom.area
            features["extent"] = areas / (width * height + 1e-6)

    if "solidity" in features_to_compute or "convexity" in features_to_compute:
        convex_hull = geom.convex_hull
        if "solidity" in features_to_compute:
            convex_areas = convex_hull.area
            if areas is None:
                areas = geom.area
            features["solidity"] = areas / (convex_areas + 1e-6)
        if "convexity" in features_to_compute:
            convex_perimeters = convex_hull.length
            if perimeters is None:
                perimeters = geom.length
            features["convexity"] = convex_perimeters / (perimeters + 1e-6)

    # Parallelized elongation and eccentricity calculation
    def compute_elong_ecc(poly):
        if not isinstance(poly, Polygon) or poly.is_empty:
            return np.nan, np.nan

        min_rect = poly.minimum_rotated_rectangle
        coords = list(min_rect.exterior.coords)
        edges = [np.linalg.norm(np.array(coords[i]) - np.array(coords[i + 1])) for i in range(4)]
        edges = sorted(edges)
        if len(edges) < 2 or edges[1] == 0:
            return np.nan, np.nan

        elongation = edges[2] / edges[1]
        a = edges[2] / 2
        b = edges[1] / 2
        eccentricity = np.sqrt(a**2 - b**2) / a if a > 0 else np.nan

        return elongation, eccentricity

    if "elongation" in features_to_compute or "eccentricity" in features_to_compute:
        results = Parallel(n_jobs=n_jobs)(delayed(compute_elong_ecc)(poly) for poly in geom)
        elongations, eccentricities = zip(*results, strict=False)
        if "elongation" in features_to_compute:
            features["elongation"] = elongations
        if "eccentricity" in features_to_compute:
            features["eccentricity"] = eccentricities

    if "compactness" in features_to_compute:
        if perimeters is None:
            perimeters = geom.length
        if areas is None:
            areas = geom.area
        features["compactness"] = (perimeters**2) / (areas + 1e-6)

    return features
