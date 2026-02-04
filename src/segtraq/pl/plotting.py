from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

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
    is_categorical = umap_df[color].dtype == "object" or pd.api.types.is_categorical_dtype(umap_df[color])

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
