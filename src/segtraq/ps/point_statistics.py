import numpy as np
import pandas as pd
import spatialdata as sd
from shapely import LinearRing, Point

from ..rc.utils import _align_expression_dfs, _assign_transcripts_to_center_or_border
from ..utils import merge_into_obs


def centroid_mean_coord_diff(
    sdata: sd.SpatialData,
    genes: str | list[str] | None = None,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    points_gene_key: str = "feature_name",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    shapes_cell_id_key: str = "cell_id",
    points_x_key: str = "x",
    points_y_key: str = "y",
    shapes_key: str = "cell_boundaries",
    centroid_key: list = ("centroid_x", "centroid_y"),
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
    shapes_cell_id_key : str or None, optional, default="cell_id"
        Column in the cell-boundary shapes linking polygons to cell IDs.
        If `None`, the shape index is used as the cell ID.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    shapes_key: str, optional
        The key in `sdata.shapes` specifying the geometry column. Default is "cell_boundaries".
    centroid_key: list, optional
        The keys to access the centroids in the `sdata.shapes` slot. Defaults are "centroid_x" and "centroid_y"
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

    # extract the transcript information
    df = sdata.points[points_key].compute()

    # filter to those cells which are in the anndata object
    df = df[df[points_cell_id_key].isin(sdata[tables_key].obs[points_cell_id_key])]

    # subset transcript dataframe to the feature
    if genes is not None:
        if isinstance(genes, str):
            df = df[df[points_gene_key] == genes]
        else:
            df = df[df[points_gene_key].isin(genes)]

        # check if the dataframe is empty after filtering
        if df.empty:
            raise ValueError(f"No transcripts found for the specified gene(s): {genes}.")

    # drop the background transcripts in cell_id == -1
    df = df[df[points_cell_id_key] != points_background_id]

    # group by cell id
    df = df.groupby(points_cell_id_key)

    # compute the mean x,y coordiantes of the transcripts per cell
    x_mean = df[points_x_key].mean()
    y_mean = df[points_y_key].mean()

    x_mean = pd.DataFrame(x_mean)
    y_mean = pd.DataFrame(y_mean)

    gdf = sdata[shapes_key].copy()

    if shapes_cell_id_key is not None:
        id_key = shapes_cell_id_key
        gdf.set_index(id_key, inplace=True)
    elif sdata[shapes_key].index.name is not None:
        id_key = sdata[shapes_key].index.name
    else:
        id_key = tables_cell_id_key
        gdf.index.name = id_key

    # extract the centroids
    df_centroids_x = pd.DataFrame(gdf.centroid.x, columns=[centroid_key[0]])
    df_centroids_y = pd.DataFrame(gdf.centroid.y, columns=[centroid_key[1]])

    # do an inner merge on the cell ids - some cells have no transcripts
    df_total_x = df_centroids_x.merge(x_mean, left_on=id_key, right_on=points_cell_id_key, how="inner")
    df_total_y = df_centroids_y.merge(y_mean, left_on=id_key, right_on=points_cell_id_key, how="inner")

    df_total = pd.concat([df_total_x, df_total_y], axis=1)

    # calculate the euclidean distance
    df_total["distance"] = np.linalg.norm(
        df_total.loc[:, [centroid_key[0], centroid_key[1]]].values
        - df_total.loc[:, [points_y_key, points_x_key]].values,
        ord=2,
        axis=1,
    )

    # extract the cell area
    area_df = sdata[tables_key].obs[[id_key, "cell_area"]]
    df_total = df_total.merge(area_df, left_on=points_cell_id_key, right_on=id_key, how="left")

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
        # removing the area, since it already exists in the object
        df_total = df_total.drop(columns=["cell_area"])
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=df_total,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=id_key,
        )

    return df_total


def distance_to_membrane(
    sdata: sd.SpatialData,
    genes: str | list[str] | None = None,
    tables_key: str = "table",
    points_gene_key: str = "feature_name",
    points_key: str = "transcripts",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    tables_cell_id_key: str = "cell_id",
    shapes_cell_id_key: str = "cell_id",
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
    shapes_cell_id_key : str or None, optional, default="cell_id"
        Column in the cell-boundary shapes linking polygons to cell IDs.
        If `None`, the shape index is used as the cell ID.
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

    # extract the transcript information
    df = sdata.points[points_key].compute()

    # filter to those cells which are in the anndata object
    df = df[df[points_cell_id_key].isin(sdata[tables_key].obs[tables_cell_id_key])]

    # subset transcript dataframe to the feature
    if genes is not None:
        if isinstance(genes, str):
            genes = [genes]
        df = df[df[points_gene_key].isin(genes)]

    # drop the background transcripts
    df = df[df[points_cell_id_key] != points_background_id]

    # zip the coordinates to a common column as tuple
    df["coordinates"] = list(zip(df[points_x_key], df[points_y_key], strict=False))

    # make the coordinates into a Point object
    df["coordinate_points"] = df["coordinates"].map(lambda x: Point(x))

    gdf = sdata[shapes_key].copy()

    if shapes_cell_id_key is not None:
        id_key = shapes_cell_id_key
        gdf.set_index(id_key, inplace=True)
    elif sdata[shapes_key].index.name is not None:
        id_key = sdata[shapes_key].index.name
    else:
        id_key = tables_cell_id_key
        gdf.index.name = id_key

    gdf = gdf.merge(df, how="inner", left_on=points_cell_id_key, right_on=id_key)

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
        feature = f"{len(genes)}_genes"
    gdf[f"distance_to_outline_{feature}"] = gdf.apply(
        lambda x: x["coordinate_points"].distance(x["linear_geometry"]), axis=1
    )

    # calculate the mean transcript distance to the cell outline per cell
    mean_distance_to_outline = gdf.groupby(id_key)[[f"distance_to_outline_{feature}"]].mean()

    # extract the cell area
    area_df = sdata[tables_key].obs[[tables_cell_id_key, "cell_area"]]
    mean_distance_to_outline = mean_distance_to_outline.merge(
        area_df, left_on=shapes_cell_id_key, right_on=tables_cell_id_key, how="left"
    )

    # normalise by area
    mean_distance_to_outline[f"distance_to_outline_inverse_{feature}"] = (
        mean_distance_to_outline[f"distance_to_outline_{feature}"] / mean_distance_to_outline["cell_area"]
    )

    # take the inverse - score is high when distance is small. sqrt transformed to handle right skewed distribution
    mean_distance_to_outline[f"distance_to_outline_inverse_{feature}"] = 1 / np.sqrt(
        mean_distance_to_outline[f"distance_to_outline_{feature}"]
    )

    mean_distance_to_outline = mean_distance_to_outline.reset_index(drop=True)

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=mean_distance_to_outline[
                [id_key, f"distance_to_outline_{feature}", f"distance_to_outline_inverse_{feature}"]
            ],
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=id_key,
        )

    return mean_distance_to_outline


def periphery_enrichment_score(
    sdata: sd.SpatialData,
    genes: str | list[str] | None = None,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    shapes_cell_id_key: str | None = "cell_id",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_gene_key: str = "feature_name",
    erosion_fraction_of_radius: float = 0.3,
    inplace: bool = True,
) -> pd.DataFrame:
    center_gdf, border_gdf, expr_center, expr_border = _assign_transcripts_to_center_or_border(
        sdata,
        shapes_key=shapes_key,
        shapes_cell_id_key=shapes_cell_id_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_gene_key=points_gene_key,
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

    # merging areas into the df
    df = df.merge(
        pd.DataFrame({"cell_id": center_gdf["cell_id"], f"center_area_{feature}": center_gdf.geometry.area.values}),
        on="cell_id",
    ).merge(
        pd.DataFrame({"cell_id": border_gdf["cell_id"], f"border_area_{feature}": border_gdf.geometry.area.values}),
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
