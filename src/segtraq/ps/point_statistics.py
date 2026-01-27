import numpy as np
import pandas as pd
import spatialdata as sd
from shapely import LinearRing, Point, Polygon

from ..rc.utils import _align_expression_dfs, _get_center_border_counts, _join_points_regions
from ..utils import filter_cells, merge_into_obs


def perc_points_outside_boundary(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    For each cell, compute the percentage of transcripts assigned to that cell
    that lie outside the cell boundary polygon.

    Uses `_join_points_regions` to spatially join all points to cell polygons, then:
      inside_assigned_cell = (region_id == points_cell_id_key)
      outside = total - inside

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned spatial
        transcriptomics data (images, tables, points, shapes and optional labels).
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons.
    points_key : str, default="transcripts"
        Key in `sdata.points` for spot/transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell (background).
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript/spot.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    inplace : bool, optional
        Whether to add the results to `sdata.tables`. Default is True.

    Returns DataFrame with:
      [points_cell_id_key, n_points_total, n_points_inside, n_points_outside, pct_points_outside]
    """

    # Join points to cell polygons (do NOT require ID match; all points required for denominator)
    pts_joined, _ = _join_points_regions(
        sdata=sdata,
        region_key=shapes_key,
        tables_key=tables_key,
        points_key=points_key,
        points_gene_key=points_gene_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        predicate="intersects",
        require_points_region_ID_match=False,
    )

    # point is within some cell polygon AND that polygon id equals its assigned cell_id
    inside_assigned = pts_joined["region_id"].notna() & (pts_joined["region_id"] == pts_joined[points_cell_id_key])

    # total points per assigned cell_id (denominator)
    total = pts_joined.groupby(points_cell_id_key, observed=True).size().rename("n_points_total")

    # inside points per assigned cell_id
    inside = pts_joined.loc[inside_assigned].groupby(points_cell_id_key, observed=True).size().rename("n_points_inside")

    out = pd.concat([total, inside], axis=1).fillna(0)
    out["n_points_total"] = out["n_points_total"].astype(int)
    out["n_points_inside"] = out["n_points_inside"].astype(int)
    out["n_points_outside"] = out["n_points_total"] - out["n_points_inside"]

    out["pct_points_outside"] = np.where(
        out["n_points_total"] > 0,
        100.0 * out["n_points_outside"] / out["n_points_total"],
        np.nan,
    )

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out[["pct_points_outside"]],
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=points_cell_id_key,
        )

    return out


def centroid_mean_coord_diff(
    sdata: sd.SpatialData,
    genes: str | list[str] | None = None,
    aggregate: bool = True,
    cell_type_key: str = "transferred_celltype_plot",
    cell_type_query: str = None,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    points_gene_key: str = "feature_name",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    shapes_key: str = "cell_boundaries",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Calculates the euclidean distance between the mean x,y coordinate of the transcripts
    indicated by the feature variable and the centroid of each cell.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial transcriptomics data.
    genes: Optional[Union[str, List[str]]] = None,
        String or list of strings indicating the feature/gene(s) to calculate the mean transcript coordiantes on.
        If None, all genes are used.
    aggregate: bool,
        Whether or not to aggregate the
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    points_gene_key : str, optional
        The key to access gene names within the transcript data. Default is "feature_name".
    points_key : str, optional
        The key in the transcript table indicating transcript identifiers. Default is "transcripts".
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        The cell ID value indicating background transcripts that should be ignored.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    shapes_key: str, optional
        The key in `sdata.shapes` specifying the geometry column. Default is "cell_boundaries".
    inplace : bool, optional
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns `["centroid_x, "centroid_y", "x", "y", "distance"]`,
        where "distance" is the euclidean distance between the coordinates `["centroid_x, "centroid_y"] and
        ["x", "y"].

    Notes
    -----
    Requires that the input AnnData table contains a "cell_area" column in `.obs`.
    """
    assert "cell_area" in sdata[tables_key].obs.columns, (
        f"'cell_area' column not found in sdata.tables['{tables_key}'].obs. "
        "Please compute cell areas before using this function, e. g. by using st.bl.morphological_features()."
    )

    # filter cells that are in query
    if cell_type_query is not None:
        adata = filter_cells(adata=sdata.tables[tables_key], col=cell_type_key, func=lambda x: x.isin(cell_type_query))
    else:
        adata = sdata.tables[tables_key]

    # extract the transcript information
    transcript_df = sdata.points[points_key].compute()

    # filter to those cells which are in the anndata object
    transcript_df = transcript_df[transcript_df[points_cell_id_key].isin(adata.obs[points_cell_id_key])]

    # subset transcript dataframe to the feature
    if genes is not None:
        if isinstance(genes, str):
            transcript_df = transcript_df[transcript_df[points_gene_key] == genes]
        else:
            transcript_df = transcript_df[transcript_df[points_gene_key].isin(genes)]

        # check if the dataframe is empty after filtering
        if transcript_df.empty:
            raise ValueError(f"No transcripts found for the specified gene(s): {genes}.")

    if aggregate:
        transcript_df[points_gene_key] = "aggregate"

    # drop the background transcripts
    transcript_df = transcript_df[transcript_df[points_cell_id_key] != points_background_id]
    # group by cell id
    transcript_df = transcript_df.groupby(points_cell_id_key)

    # compute the mean x, y coordinates of the transcripts per cell
    x_mean = transcript_df[points_x_key].mean()
    y_mean = transcript_df[points_y_key].mean()
    x_mean = pd.DataFrame(x_mean)
    y_mean = pd.DataFrame(y_mean)

    gdf = sdata[shapes_key].copy()

    # extract the centroids
    # it is important here that the centroid keys are not identical to the points_x_key and points_y_key
    centroid_key = [f"{points_x_key}_cell", f"{points_y_key}_cell"]
    df_centroids_x = pd.DataFrame(gdf.centroid.x, columns=[centroid_key[0]])
    df_centroids_y = pd.DataFrame(gdf.centroid.y, columns=[centroid_key[1]])

    # do an inner merge on the cell ids - some cells have no transcripts
    df_total_x = df_centroids_x.merge(x_mean, left_on=gdf.index.name, right_on=points_cell_id_key, how="inner")
    df_total_y = df_centroids_y.merge(y_mean, left_on=gdf.index.name, right_on=points_cell_id_key, how="inner")

    df_total = pd.concat([df_total_x, df_total_y], axis=1)

    # calculate the euclidean distance
    df_total["distance"] = np.linalg.norm(
        df_total.loc[:, [centroid_key[0], centroid_key[1]]].values
        - df_total.loc[:, [points_y_key, points_x_key]].values,
        ord=2,
        axis=1,
    )

    # extract the cell area
    area_df = sdata[tables_key].obs[[gdf.index.name, "cell_area"]]
    df_total = df_total.merge(area_df, left_on=points_cell_id_key, right_on=gdf.index.name, how="left")

    # normalise the cell area
    if genes is None:
        feature = "all_genes"
    elif isinstance(genes, str):
        feature = genes
    else:
        feature = f"{len(genes)}_genes"
    df_total[f"distance_{feature}"] = df_total["distance"] / df_total["cell_area"]
    df_total = df_total.reset_index(drop=True)

    if inplace:
        # only keep new, relevant columns
        df_total = df_total[[gdf.index.name, f"distance_{feature}"]]
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=df_total,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=gdf.index.name,
        )

    return df_total


def distance_to_membrane(
    sdata: sd.SpatialData,
    genes: str | list[str] | None = None,
    aggregate: bool = True,
    cell_type_key: str = "transferred_celltype_plot",
    cell_type_query: str = None,
    tables_key: str = "table",
    points_gene_key: str = "feature_name",
    points_key: str = "transcripts",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    inplace: bool = True,
):
    """
    Calculates the mean distance of the transcript of a feature of interest to the outline of the cell segmentation

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial transcriptomics data.
    genes: str | list[str] | None = None,
        String or list of strings indicating the feature/gene(s) to calculate the mean transcript distances on.
        If None, all genes are used.
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    points_gene_key : str, optional
        The key to access gene names within the transcript data. Default is "feature_name".
    points_key : str, optional
        The key in the transcript table indicating transcript identifiers. Default is "transcripts".
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    points_background_id: str | int = "UNASSIGNED"
        The cell ID value indicating background transcripts that should be ignored.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    shapes_key: str, optional
        The key in `sdata.shapes` specifying the geometry column. Default is "cell_boundaries".
    inplace : bool, optional
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns `["distance_to_outline_inverse", f"distance_to_outline_{feature}" and "cell_area"]`

    Notes
    -----
    Requires that the input AnnData table contains a "cell_area" column in `.obs`.

    """
    assert "cell_area" in sdata[tables_key].obs.columns, (
        f"'cell_area' column not found in sdata.tables['{tables_key}'].obs. "
        "Please compute cell areas before using this function, e. g. by using st.bl.morphological_features()."
    )

    # filter cells that are in query
    if cell_type_query is not None:
        adata = filter_cells(adata=sdata.tables[tables_key], col=cell_type_key, func=lambda x: x.isin(cell_type_query))
    else:
        adata = sdata.tables[tables_key]

    # extract the transcript information
    transcript_df = sdata.points[points_key].compute()

    # filter to those cells which are in the anndata object
    transcript_df = transcript_df[transcript_df[points_cell_id_key].isin(adata.obs[tables_cell_id_key])]

    # subset transcript dataframe to the feature
    if genes is not None:
        if isinstance(genes, str):
            genes = [genes]
        transcript_df = transcript_df[transcript_df[points_gene_key].isin(genes)]

    if aggregate:
        transcript_df[points_gene_key] = "aggregate"

    # drop the background transcripts
    transcript_df = transcript_df[transcript_df[points_cell_id_key] != points_background_id]

    # zip the coordinates to a common column as tuple
    transcript_df["coordinates"] = list(zip(transcript_df[points_x_key], transcript_df[points_y_key], strict=False))

    # make the coordinates into a Point object
    transcript_df["coordinate_points"] = transcript_df["coordinates"].map(lambda x: Point(x))

    gdf = sdata[shapes_key].copy()

    gdf = gdf.merge(transcript_df, how="inner", left_index=True, right_on=points_cell_id_key)
    gdf = gdf.set_index(points_cell_id_key)
    gdf = gdf.explode()

    # compute the linear outline of the cell segmentation
    gdf["linear_geometry"] = gdf.apply(lambda x: LinearRing(x["geometry"].exterior.coords), axis=1)

    # drop NaN values in the coordinate point column
    gdf = gdf.dropna(subset="coordinate_points")

    # calculate the distance of the transcript points to the linear segment
    if genes is None:
        feature = "all_genes"
    elif len(genes) == 1:
        feature = genes[0]
    else:
        if aggregate:
            feature = "aggregated_genes"
        else:
            feature = f"{len(genes)}_genes"

    # check whether the coordinate points are within the linear geometry
    # potentially overkill if transcript id is checked
    gdf["is_within"] = gdf.apply(lambda x: Polygon(x["linear_geometry"]).contains(x["coordinate_points"]), axis=1)

    # Then calculate distance only for inside points
    gdf[f"distance_to_outline_{feature}"] = gdf.apply(
        lambda x: x["coordinate_points"].distance(x["linear_geometry"]) if x["is_within"] else None, axis=1
    )

    # calculate the mean transcript distance to the cell outline per cell
    mean_distance_to_outline = gdf.groupby(gdf.index.name)[[f"distance_to_outline_{feature}"]].mean()

    # extract the cell area
    area_df = sdata[tables_key].obs[[tables_cell_id_key, "cell_area"]]
    mean_distance_to_outline = mean_distance_to_outline.merge(
        area_df, left_on=gdf.index.name, right_on=tables_cell_id_key, how="left"
    )

    # normalise by area
    mean_distance_to_outline[f"distance_to_outline_norm_{feature}"] = (
        mean_distance_to_outline[f"distance_to_outline_{feature}"] / mean_distance_to_outline["cell_area"]
    )

    # take the inverse - score is high when distance is small. sqrt transformed to handle right skewed distribution
    mean_distance_to_outline[f"distance_to_outline_inverse_{feature}"] = 1 / np.sqrt(
        mean_distance_to_outline[f"distance_to_outline_{feature}"]
    )

    mean_distance_to_outline = mean_distance_to_outline.reset_index(drop=True)

    if inplace:
        # only keep new, relevant columns
        mean_distance_to_outline = mean_distance_to_outline[
            [
                gdf.index.name,
                f"distance_to_outline_{feature}",
                f"distance_to_outline_norm_{feature}",
                f"distance_to_outline_inverse_{feature}",
            ]
        ]
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=mean_distance_to_outline,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=gdf.index.name,
        )

    return mean_distance_to_outline


def periphery_enrichment_score(
    sdata: sd.SpatialData,
    genes: str | list[str] | None = None,
    aggregate: bool = True,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_gene_key: str = "feature_name",
    erosion_fraction_of_radius: float = 0.3,
    inplace: bool = True,
) -> pd.DataFrame:
    expr_center, expr_border = _get_center_border_counts(
        sdata,
        tables_key=tables_key,
        shapes_key=shapes_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_gene_key=points_gene_key,
        points_background_id=points_background_id,
        erosion_fraction_of_radius=erosion_fraction_of_radius,
    )

    if genes is None:
        feature = "all_genes"
    elif isinstance(genes, str):
        feature = genes
    else:
        feature = f"{len(genes)}_genes"

    # next, we align the three expression DataFrames to have the same cells and genes
    aligned_expression_dfs = _align_expression_dfs(
        {f"expr_center_{feature}": expr_center, f"expr_border_{feature}": expr_border}, sdata, tables_key
    )

    expr_center_raw = aligned_expression_dfs[f"expr_center_{feature}"]
    expr_border_raw = aligned_expression_dfs[f"expr_border_{feature}"]
    # summing up the total expression per cell (only on the selected genes)
    if genes is not None:
        if isinstance(genes, str):
            genes = [genes]
        expr_center_raw = expr_center_raw[genes]
        expr_border_raw = expr_border_raw[genes]
        assert not expr_center_raw.empty, "No transcripts found for the specified gene(s) in the center region."
        assert not expr_border_raw.empty, "No transcripts found for the specified gene(s) in the border region."

    total_expr_center = expr_center_raw.sum(axis=1)
    total_expr_border = expr_border_raw.sum(axis=1)

    # combining the areas and expressions into a single DataFrame
    epsilon = 1e-10  # small constant to avoid division by zero

    # combining expressions
    df = pd.DataFrame(
        {
            "cell_id": total_expr_center.index,
            f"center_expr_{feature}": total_expr_center.values,
            f"border_expr_{feature}": total_expr_border.values,
        }
    )

    center_gdf = sdata.shapes["cell_centers"]
    border_gdf = sdata.shapes["cell_borders"]

    # merging areas into the df
    df = df.merge(
        pd.DataFrame({"cell_id": center_gdf.index, f"center_area_{feature}": center_gdf.geometry.area.values}),
        on="cell_id",
    ).merge(
        pd.DataFrame({"cell_id": border_gdf.index, f"border_area_{feature}": border_gdf.geometry.area.values}),
        on="cell_id",
    )

    # calculate densities and ratio with pseudocount (+1) and safe division (+epsilon)
    # note that we set density_ratio to NaN if both center and border expression are zero
    df[f"border_density_{feature}"] = (df[f"border_expr_{feature}"] + 1) / (df[f"border_area_{feature}"] + epsilon)
    df[f"center_density_{feature}"] = (df[f"center_expr_{feature}"] + 1) / (df[f"center_area_{feature}"] + epsilon)
    df[f"density_ratio_{feature}"] = df[f"border_density_{feature}"] / df[f"center_density_{feature}"]
    mask = (df[f"border_expr_{feature}"] + df[f"center_expr_{feature}"]) > 0
    df.loc[~mask, f"density_ratio_{feature}"] = np.nan
    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=df,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key="cell_id",
        )

    return df
