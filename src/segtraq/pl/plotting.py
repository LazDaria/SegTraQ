from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spatialdata as sd
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d

from .utils import (
    build_celltype_composition_df,
    build_obs_box_df,
    build_umap_df,
)


def celltype_proportions(
    method_to_segtraq: dict[str, object],
    celltype_col: str,
    ct_palette: Mapping[str, str] | None = None,
    title: str = "Cell-type proportions",
    table_key: str = "table",
    include_zeros: bool = True,
    missing_label: str = "None",
    save: str | None = None,
) -> pd.DataFrame:
    """
    Plots a stacked barplot of cell type proportions per segmentation method.

    Parameters
    ----------
    method_to_segtraq : dict[str, object]
        A dictionary mapping segmentation method names to SegTraQ objects.
    celltype_col : str
        The column name in `adata.obs` that contains cell type labels.
    ct_palette : Mapping[str, str] | None, optional
        A mapping from cell type names to colors. If None, a default palette is used.
    title : str, optional
        The title of the plot.
    table_key : str, optional
        The key to access the AnnData table in the SegTraQ object.
    include_zeros : bool, optional
        Whether to include cell types with zero counts in the plot.
    missing_label : str, optional
        The label to use for missing cell type annotations.
    save : str | None, optional
        If provided, the path to save the plot. If None, the plot is shown.
    """
    # converting the dictionary of SegTraQ objects to a dictionary of AnnData objects
    method_to_adata = {method: stq.sdata.tables[table_key] for method, stq in method_to_segtraq.items()}

    # creating a dataframe that contains cell type proportions per method
    comp_df = build_celltype_composition_df(method_to_adata, celltype_col, include_zeros, missing_label)

    methods = comp_df["Segmentation Method"].unique().tolist()
    celltypes = sorted(comp_df["Cell Type"].unique().tolist(), key=str)

    # creating a palette, either from the one provided or a default
    if ct_palette is None:
        ct_palette = {ct: plt.get_cmap("tab20")(i % 20) for i, ct in enumerate(celltypes)}
    color_list = [ct_palette.get(ct, "#aaaaaa") for ct in celltypes]

    # reshape to (celltype x method) for proportions
    pivot = comp_df.pivot_table(
        index="Cell Type", columns="Segmentation Method", values="Proportion", fill_value=0.0
    ).reindex(celltypes)

    x = np.arange(len(methods))
    width = 0.7
    bottoms = np.zeros(len(methods), dtype=float)

    fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(methods)), 5))
    for ct, color in zip(celltypes, color_list, strict=False):
        heights = np.array([pivot.loc[ct, m] if m in pivot.columns else 0.0 for m in methods], dtype=float)
        ax.bar(x, heights, width, bottom=bottoms, color=color, edgecolor="white", label=ct)
        bottoms += heights  # stack (Matplotlib stacking uses 'bottom')  :contentReference[oaicite:1]{index=1}

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.legend(title="Cell Type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()

    if save is not None:
        output_path = Path(save)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    # also return/save counts table for convenience
    counts_df = comp_df[["Segmentation Method", "Cell Type", "Count"]].copy()
    return counts_df


def umap(
    method_to_segtraq: dict[str, object],
    color: str,
    palette: Mapping[str, str] | None = None,
    table_key: str = "table",
    umap_key: str = "X_umap",
    point_size: float = 6.0,
    figsize: tuple[float, float] | None = None,
    cols: int = 3,
    legend: bool = False,
    cmap: str = "viridis",
    save: str | None = None,
) -> pd.DataFrame:
    """
    UMAP scatter plots per segmentation method, colored by a specified column.
    One global legend or colorbar is used.
    """
    # convert SegTraQ → AnnData
    method_to_adata = {method: stq.sdata.tables[table_key] for method, stq in method_to_segtraq.items()}

    # build UMAP dataframe
    umap_df = build_umap_df(
        method_to_adata,
        feature_col=color,
        umap_key=umap_key,
    )

    # validate color column
    if color not in umap_df.columns:
        raise ValueError(f"Column '{color}' not found in UMAP dataframe. Available columns: {list(umap_df.columns)}")

    methods = umap_df["Segmentation Method"].unique().tolist()
    n = len(methods)
    rows = int(np.ceil(n / cols))

    if figsize is None:
        figsize = (6 * cols, 7 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    # determine feature type
    is_categorical = umap_df[color].dtype == "object" or isinstance(umap_df[color].dtype, pd.CategoricalDtype)

    # categorical palette handling
    if is_categorical:
        values = sorted(umap_df[color].dropna().unique().tolist())
        if palette is None:
            palette = {v: plt.get_cmap("tab20")(i % 20) for i, v in enumerate(values)}
        else:
            # ensure all categories are covered
            palette = {v: palette.get(v, "#aaaaaa") for v in values}
    else:
        # continuous: define a shared normalization
        vmin, vmax = umap_df[color].min(), umap_df[color].max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # plot each method
    for i, method in enumerate(methods):
        ax = axes[i // cols, i % cols]
        df_m = umap_df[umap_df["Segmentation Method"] == method]

        if is_categorical:
            sns.scatterplot(
                data=df_m,
                x="x",
                y="y",
                hue=color,
                palette=palette,
                s=point_size,
                linewidth=0,
                ax=ax,
                legend=False,  # suppress per-axis legend
            )
        else:
            sc = ax.scatter(
                df_m["x"],
                df_m["y"],
                c=df_m[color],
                cmap=cmap,
                s=point_size,
                linewidth=0,
                norm=norm,
            )

        ax.set_title(method)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    # hide unused axes
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    # global legend / colorbar
    if legend:
        if is_categorical:
            handles = [Patch(color=palette[v], label=v) for v in palette]
            fig.legend(
                handles=handles,
                title=color,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=False,
            )
        else:
            # cbar for continuous
            _ = fig.colorbar(sc, ax=axes, label=color, shrink=0.8)

    fig.suptitle(f"UMAP colored by {color}", y=0.995, fontsize=14)
    # we do not use tight_layout here to avoid messing with the position of the legend/colorbar

    if save is not None:
        output_path = Path(save)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return umap_df


def boxplot_combined(
    method_to_segtraq: dict[str, object],
    celltype_col: str,
    value_key: str,
    method_palette: Mapping[str, str] | None = None,
    table_key: str = "table",
    x_order: list[str] | None = None,
    title: str | None = None,
    dropna: bool = True,
    missing_label: str = "None",
    save: str | None = None,
) -> pd.DataFrame:
    """
    Combined boxplot with all methods overlaid, with cell types on the x-axis.

    Parameters
    ----------
    method_to_segtraq : dict[str, object]
        A dictionary mapping segmentation method names to SegTraQ objects.
    celltype_col : str
        The column name in `adata.obs` that contains cell type labels.
    value_key : str
        The column name in `adata.obs` that contains the values to plot.
    method_palette : Mapping[str, str] | None, optional
        A mapping from segmentation method names to colors. If None, a default palette is used.
    table_key : str, optional
        The key to access the AnnData table in the SegTraQ object.
    x_order : list[str] | None, optional
        The order of cell types on the x-axis. If None, cell types are sorted alphabetically.
    title : str | None, optional
        The title of the plot. If None, a default title is used.
    dropna : bool, optional
        Whether to drop NaN values in the value column.
    missing_label : str, optional
        The label to use for missing cell type annotations.
    save : str | None, optional
        If provided, the path to save the plot. If None, the plot is shown.
    """
    method_to_adata = {method: stq.sdata.tables[table_key] for method, stq in method_to_segtraq.items()}

    box_df = build_obs_box_df(method_to_adata, celltype_col, value_key, dropna, missing_label)

    if box_df.empty:
        raise ValueError("box_df is empty.")

    if x_order is None:
        x_order = sorted(box_df["Cell Type"].unique().tolist())

    value_key = box_df["variable"].iloc[0] if "variable" in box_df.columns else "value"

    if method_palette is None:
        methods = box_df["Segmentation Method"].unique().tolist()
        method_palette = {m: plt.get_cmap("tab10")(i % 10) for i, m in enumerate(methods)}

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(x_order)), 5))

    sns.boxplot(
        data=box_df,
        x="Cell Type",
        y="value",
        hue="Segmentation Method",
        order=x_order,
        palette=method_palette,
        ax=ax,
    )  # grouped boxplot via hue :contentReference[oaicite:3]{index=3}

    ax.set_xlabel("Cell Type")
    ax.set_ylabel(value_key)
    ax.set_title(title or f"{value_key} by Cell Type")
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    if save is not None:
        output_path = Path(save)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    # return the dataframe for convenience
    return box_df


def boxplot(
    method_to_segtraq: dict[str, object],
    celltype_col: str,
    value_key: str,
    method_palette: Mapping[str, str] | None = None,
    x_order: list[str] | None = None,
    table_key: str = "table",
    title: str | None = None,
    dropna: bool = True,
    missing_label: str = "None",
    save: str | None = None,
) -> pd.DataFrame:
    """
    Boxplots per segmentation method, with cell types on the x-axis.

    Parameters
    ----------
    method_to_segtraq : dict[str, object]
        A dictionary mapping segmentation method names to SegTraQ objects.
    celltype_col : str
        The column name in `adata.obs` that contains cell type labels.
    value_key : str
        The column name in `adata.obs` that contains the values to plot.
    method_palette : Mapping[str, str] | None, optional
        A mapping from segmentation method names to colors. If None, a default palette is used.
    x_order : list[str] | None, optional
        The order of cell types on the x-axis. If None, cell types are sorted alphabetically.
    table_key : str, optional
        The key to access the AnnData table in the SegTraQ object.
    title : str | None, optional
        The title of the plot. If None, no title is set.
    dropna : bool, optional
        Whether to drop NaN values in the value column.
    missing_label : str, optional
        The label to use for missing cell type annotations.
    save : str | None, optional
        If provided, the path to save the plot. If None, the plot is shown.
    """
    method_to_adata = {method: stq.sdata.tables[table_key] for method, stq in method_to_segtraq.items()}

    box_df = build_obs_box_df(method_to_adata, celltype_col, value_key, dropna, missing_label)
    if box_df.empty:
        raise ValueError("box_df is empty.")

    if x_order is None:
        x_order = sorted(box_df["Cell Type"].unique().tolist())

    value_key = box_df["variable"].iloc[0] if "variable" in box_df.columns else "value"

    method_order = [m for m in method_to_adata.keys() if m in box_df["Segmentation Method"].unique()]
    method_order += [m for m in box_df["Segmentation Method"].unique() if m not in method_order]
    n_methods = len(method_order)
    if n_methods == 0:
        raise ValueError("No methods found in box_df.")

    if method_palette is None:
        method_palette = {m: plt.get_cmap("tab10")(i % 10) for i, m in enumerate(method_order)}

    fig_width = max(10, 0.8 * len(x_order))
    height_per_row = 4
    fig, axes = plt.subplots(
        nrows=n_methods,
        ncols=1,
        figsize=(fig_width, height_per_row * n_methods),
        sharex=True,  # shared categories across panels
        sharey=False,  # independent y-axes
    )
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, method_order, strict=False):
        df_m = box_df[box_df["Segmentation Method"] == method]
        if df_m.empty:
            ax.axis("off")
            ax.set_title(f"{method} (no data)")
            continue

        color = method_palette.get(method, None) if method_palette is not None else None

        sns.boxplot(data=df_m, x="Cell Type", y="value", order=x_order, color=color, ax=ax)

        ax.set_ylabel(value_key)
        ax.set_title(method)

    # Hide x tick labels on all but the bottom axis *without* clearing shared labels
    for ax in axes[:-1]:
        ax.tick_params(axis="x", which="both", labelbottom=False)
        ax.set_xlabel("")

    # Format bottom axis labels
    axes[-1].set_xlabel("Cell Type")
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")

    if title:
        fig.suptitle(title, y=0.995)
        fig.subplots_adjust(top=0.93)

    fig.tight_layout()

    if save is not None:
        output_path = Path(save)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    # return the dataframe for convenience
    return box_df


def transcript_distribution_across_space(
    sdata: sd.SpatialData,
    axes: str | tuple[str] | list[str] = ("x", "y"),
    filter_size: int = 21,
    points_key: str = "transcripts",
) -> plt.Axes | list[plt.Axes]:
    """
    Plot the marginal distribution of transcripts along one or more spatial axes.

    Parameters
    ----------
    sdata : sd.SpatialData
        A SpatialData object containing a points element with transcript coordinates.
    axes : str or list/tuple of str, optional
        Spatial axis or axes to plot. Default is ('x', 'y').
    filter_size : int, optional
        Size of the filter kernel. Default is 21.
    points_key : str, optional
        Key inside ``sdata.points`` that holds the transcript DataFrame.
        Default is ``'transcripts'``.

    Returns
    -------
    matplotlib.axes.Axes or list of matplotlib.axes.Axes
        The Axes containing the plot(s). Returns a single Axes if a single
        axis string was passed, otherwise a list.

    Raises
    ------
    ValueError
        If any of the requested axes are not present in the points DataFrame.
    """
    # ── 0. Handle inputs ─────────────────────────────────────────────────────
    single = isinstance(axes, str)
    axes = [axes] if single else list(axes)

    # ── 1. Validate axes ─────────────────────────────────────────────────────
    points_df = sdata.points[points_key]
    available = list(points_df.columns)
    missing = [a for a in axes if a not in available]
    if missing:
        raise ValueError(
            f"Requested axis/axes {missing} not found in '{points_key}'. "
            f"Available columns are: {available}. Please set the axes parameter accordingly."
        )

    # ── 2. Create subplots ───────────────────────────────────────────────────
    n = len(axes)
    fig, ax_list = plt.subplots(n, 1, figsize=(9, 4 * n), squeeze=False)
    ax_list = ax_list[:, 0]  # shape (n,)

    # ── 3. Plot each axis ────────────────────────────────────────────────────
    assert filter_size > 0, "Filter size must be positive."
    assert filter_size % 2 == 1, "Filter size should be odd for symmetric smoothing."

    for ax, axis in zip(ax_list, axes, strict=False):
        coords = points_df[axis].compute()

        n_bins = int(coords.max() - coords.min()) + 1
        counts, bin_edges = np.histogram(coords, bins=n_bins)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        smoothed = uniform_filter1d(counts.astype(float), size=filter_size)

        ax.plot(bin_centres, smoothed, color="steelblue", linewidth=1.4, label=f"Smoothed (filter={filter_size})")
        ax.set_xlabel(axis)
        ax.set_ylabel("Transcript count")
        ax.set_title(f"Transcript distribution along {axis}")
        ax.legend(fontsize=8)
        ax.set_xlim(bin_centres[0], bin_centres[-1])

    plt.tight_layout()
    return ax_list[0] if single else list(ax_list)


def feature_distribution_across_space(
    sdata: sd.SpatialData,
    features: str | tuple[str] | list[str],
    axes: tuple[str] | list[str] = ("centroid_x", "centroid_y"),
    filter_size: int = 21,
    tables_key: str = "table",
) -> list[list[plt.Axes]]:
    """
    Plot the distribution of one or more obs features across spatial axes.

    For each feature, one subplot per axis is shown, arranged as a grid of
    shape (n_features, n_axes).

    Parameters
    ----------
    sdata : sd.SpatialData
        A SpatialData object containing an AnnData table.
    features : str or list/tuple of str
        One or more column names from ``adata.obs`` to plot.
    axes : list/tuple of str, optional
        Columns in ``adata.obs`` to use as spatial axes. Default is
        ``('centroid_x', 'centroid_y')``.
    filter_size : int, optional
        Size of the filter kernel. Default is 21.
    tables_key : str, optional
        Key inside ``sdata.tables``. Default is ``'table'``.

    Returns
    -------
    list of list of matplotlib.axes.Axes
        Nested list of shape [n_features][n_axes].

    Raises
    ------
    ValueError
        If any requested feature or axis is not found in obs.
    """
    # ── 0. Handle inputs ─────────────────────────────────────────────────────
    features = [features] if isinstance(features, str) else list(features)
    axes = list(axes)

    # ── 1. Validate ──────────────────────────────────────────────────────────
    adata = sdata.tables[tables_key]
    obs_cols = list(adata.obs.columns)

    missing_axes = [a for a in axes if a not in obs_cols]
    if missing_axes:
        raise ValueError(
            f"Axis column(s) {missing_axes} not found in obs. "
            f"Available columns are: {obs_cols}. "
            f"Please set the axes parameter accordingly."
        )

    missing_features = [f for f in features if f not in obs_cols]
    if missing_features:
        raise ValueError(
            f"Feature(s) {missing_features} not found in obs. "
            f"Available columns are: {obs_cols}. "
            f"Please set the features parameter accordingly."
        )

    # we want to only plot numeric features,
    # but we allow automatic conversion of non-numeric columns that can be converted to numeric
    # (e.g. categorical columns with numeric categories, such as FOV labels '1', '2', '3' etc.)
    non_numeric = [f for f in features if not pd.api.types.is_numeric_dtype(adata.obs[f])]

    if non_numeric:
        convertible = []
        non_convertible = []
        for f in non_numeric:
            try:
                pd.to_numeric(adata.obs[f])
                convertible.append(f)
            except (ValueError, TypeError):
                non_convertible.append(f)

        if non_convertible:
            raise ValueError(
                f"Feature(s) {non_convertible} are not numeric and could not be converted. "
                f"Please provide only numerical features."
            )
        if convertible:
            for f in convertible:
                adata.obs[f] = pd.to_numeric(adata.obs[f])

    # ── 2. Create subplot grid (n_features rows × n_axes cols) ───────────────
    n_feat = len(features)
    n_axes = len(axes)
    fig, ax_grid = plt.subplots(
        n_feat,
        n_axes,
        figsize=(6 * n_axes, 4 * n_feat),
        squeeze=False,
    )

    assert filter_size > 0, "Filter size must be positive."
    assert filter_size % 2 == 1, "Filter size should be odd for symmetric smoothing."

    # ── 3. Fill grid ─────────────────────────────────────────────────────────
    for row, feature in enumerate(features):
        feature_vals = adata.obs[feature].to_numpy(dtype=float)

        for col, axis in enumerate(axes):
            ax = ax_grid[row, col]
            coords = adata.obs[axis].to_numpy(dtype=float)

            # Drop NaN centroids
            valid_mask = ~np.isnan(coords)
            n_invalid = (~valid_mask).sum()
            if n_invalid > 0:
                warnings.warn(
                    f"{n_invalid}/{len(coords)} cells have NaN coordinates in "
                    f"'{axis}' and will be excluded from plotting.",
                    stacklevel=2,
                )
                coords = coords[valid_mask]
                feature_vals_ax = feature_vals[valid_mask]
            else:
                feature_vals_ax = feature_vals

            n_bins = int(coords.max() - coords.min()) + 1
            bin_edges = np.linspace(coords.min(), coords.max(), n_bins + 1)
            bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            bin_indices = np.clip(np.digitize(coords, bin_edges[:-1]) - 1, 0, n_bins - 1)
            bin_sums = np.bincount(bin_indices, weights=feature_vals_ax, minlength=n_bins)
            bin_counts = np.bincount(bin_indices, minlength=n_bins)

            with np.errstate(invalid="ignore"):
                bin_means = np.where(bin_counts > 0, bin_sums / bin_counts, np.nan)

            nans = np.isnan(bin_means)
            if nans.any():
                bin_means[nans] = np.interp(
                    np.flatnonzero(nans),
                    np.flatnonzero(~nans),
                    bin_means[~nans],
                )

            smoothed = uniform_filter1d(bin_means, size=filter_size)

            ax.plot(bin_centres, smoothed, color="steelblue", linewidth=1.4, label=f"Smoothed (filter={filter_size})")
            ax.set_xlabel(axis)
            ax.set_ylabel(feature)
            ax.set_title(f"{feature} along {axis}")
            ax.legend(fontsize=8)
            ax.set_xlim(bin_centres[0], bin_centres[-1])

    plt.tight_layout()
    return ax_grid.tolist()
