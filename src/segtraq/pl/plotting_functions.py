from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

# =========================
# BUILDERS (method-agnostic)
# =========================


def _fill_celltype(s: pd.Series, missing_label: str = "None") -> pd.Series:
    # Categorical needs category added before fillna; otherwise cast to string then fill.
    if pd.api.types.is_categorical_dtype(s):
        if missing_label not in s.cat.categories:
            s = s.cat.add_categories([missing_label])
        return s.fillna(missing_label)
    else:
        return s.astype("string").fillna(missing_label)


def build_celltype_composition_df(
    method_to_adata: dict[str, ad.AnnData],
    celltype_col: str,
    include_zeros: bool = True,
    missing_label: str = "None",
) -> pd.DataFrame:
    frames = []
    for method, adata in method_to_adata.items():
        ct = _fill_celltype(adata.obs[celltype_col], missing_label)
        vc = ct.value_counts(dropna=False)  # counts
        props = ct.value_counts(normalize=True, dropna=False)  # proportions
        df = (
            pd.DataFrame({"Count": vc, "Proportion": props})
            .rename_axis("Cell Type")
            .reset_index()
            .assign(**{"Segmentation Method": method})
        )
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    if include_zeros:
        methods = sorted(method_to_adata.keys())
        celltypes = sorted(out["Cell Type"].unique().tolist())
        full = pd.MultiIndex.from_product([methods, celltypes], names=["Segmentation Method", "Cell Type"]).to_frame(
            index=False
        )
        out = full.merge(out, how="left", on=["Segmentation Method", "Cell Type"]).fillna(
            {"Count": 0, "Proportion": 0.0}
        )
    return out


def build_umap_and_scores_df(
    method_to_adata: dict[str, ad.AnnData],
    celltype_col: str,
    umap_key: str = "X_umap",
    bl_metrics_path: tuple[str, str] = ("segtraq", "bl", "summary"),
    cs_metrics_path: tuple[str, str] = ("segtraq", "cs"),
    missing_label: str = "None",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, mrows = [], []
    for method, adata in method_to_adata.items():
        if umap_key not in adata.obsm:
            raise KeyError(f"{method}: adata.obsm['{umap_key}'] not found.")
        umap = np.asarray(adata.obsm[umap_key])
        if umap.shape[1] != 2:
            raise ValueError(f"{method}: {umap_key} must be n_cells x 2.")

        ct = _fill_celltype(adata.obs[celltype_col], missing_label)
        rows.append(
            pd.DataFrame(
                {
                    "x": umap[:, 0],
                    "y": umap[:, 1],
                    "Cell Type": ct.values,
                    "Segmentation Method": method,
                }
            )
        )

        d = adata.uns
        cs = d
        for k in cs_metrics_path:
            cs = cs.get(k, {}) if isinstance(cs, dict) else {}

        bl = d
        for k in bl_metrics_path:
            bl = bl.get(k, {}) if isinstance(bl, dict) else {}

        mrows.append(
            {
                "Segmentation Method": method,
                "n_cells": bl.get("num_cells", np.nan),
                "perc_unassigned": bl.get("perc_unassigned_transcripts", np.nan),
                "rmsd": cs.get("rmsd", np.nan),
                "silhouette": cs.get("silhouette", np.nan),
                "ari": cs.get("ari", np.nan),
                "purity": cs.get("purity", np.nan),
            }
        )

    return pd.concat(rows, ignore_index=True), pd.DataFrame(mrows)


def build_obs_box_df(
    method_to_adata: dict[str, ad.AnnData],
    celltype_col: str,
    value_key: str,
    dropna: bool = True,
    missing_label: str = "None",
) -> pd.DataFrame:
    frames = []
    for method, adata in method_to_adata.items():
        if value_key not in adata.obs:
            continue
        ct = _fill_celltype(adata.obs[celltype_col], missing_label)
        d = pd.DataFrame({"Cell Type": ct, "value": adata.obs[value_key]})
        d["Segmentation Method"] = method
        d["variable"] = value_key
        if dropna:
            d = d.dropna(subset=["value"])
        frames.append(d)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["Segmentation Method", "Cell Type", "value", "variable"])
    )


def build_mecr_df(method_to_mecr: Mapping[str, Mapping[tuple[str, str], float]]) -> pd.DataFrame:
    """
    Flatten {(gene1,gene2)->MECR} dicts for many methods into one DF.

    Columns: ['Segmentation Method','gene1','gene2','MECR']
    """
    frames = []
    for method, mecr in method_to_mecr.items():
        if not mecr:
            continue
        part = pd.DataFrame(
            [(g1, g2, float(v)) for (g1, g2), v in mecr.items()],
            columns=["gene1", "gene2", "MECR"],
        )
        part["Segmentation Method"] = method
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["Segmentation Method", "gene1", "gene2", "MECR"])
    return pd.concat(frames, ignore_index=True)


# =========================
# PLOTS (method-agnostic)
# =========================


def plot_celltype_proportions_stacked(
    method_to_adata: dict[str, ad.AnnData],
    celltype_col: str,
    ct_palette: Mapping[str, str],
    output_path: Path,
    filename: str = "celltype_proportions_stacked.pdf",
    title: str = "Cell-type proportions",
    include_zeros: bool = True,
    missing_label: str = "None",
) -> pd.DataFrame:
    """
    Stacked bar plot (one bar per method) of cell-type proportions.

    Expects comp_df from build_celltype_composition_df.
    """

    comp_df = build_celltype_composition_df(method_to_adata, celltype_col, include_zeros, missing_label)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    methods = comp_df["Segmentation Method"].unique().tolist()
    celltypes = sorted(comp_df["Cell Type"].unique().tolist(), key=str)
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
    # fig.savefig(output_path / filename, bbox_inches="tight")
    fig.savefig(output_path / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # also return/save counts table for convenience
    counts_df = comp_df[["Segmentation Method", "Cell Type", "Count"]].copy()
    counts_df.to_csv(output_path / "celltype_counts_per_method.csv", index=False)
    return counts_df


def plot_umaps_by_feature(
    method_to_adata: dict[str, ad.AnnData],
    celltype_col: str,
    ct_palette: Mapping[str, str],
    output_path: Path,
    umap_key: str = "X_umap",
    bl_metrics_path: tuple[str, str] = ("segtraq", "bl", "summary"),
    cs_metrics_path: tuple[str, str] = ("segtraq", "cs"),
    missing_label: str = "None",
    filename: str = "umap_by_celltype_all_methods.pdf",
    point_size: float = 6.0,
    cols: int = 3,
    show_legend: bool = False,
) -> None:
    """
    Side-by-side UMAPs: one panel per method, colored by cell type.

    Expects:
      - umap_df from build_umap_and_scores_df
      - scores_df from build_umap_and_scores_df
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    umap_df, scores_df = build_umap_and_scores_df(
        method_to_adata, celltype_col, umap_key, bl_metrics_path, cs_metrics_path, missing_label
    )

    methods = umap_df["Segmentation Method"].unique().tolist()
    n = len(methods)
    cols = max(1, cols)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 7 * rows), squeeze=False)

    all_celltypes = sorted(umap_df["Cell Type"].dropna().unique().tolist())
    palette = {ct: ct_palette.get(ct, "#aaaaaa") for ct in all_celltypes}

    # draw each method in its own axis
    for i, method in enumerate(methods):
        ax = axes[i // cols, i % cols]
        df_m = umap_df[umap_df["Segmentation Method"] == method]
        # Use seaborn scatter with categorical hue (palette mapping)  :contentReference[oaicite:2]{index=2}
        sns.scatterplot(
            data=df_m, x="x", y="y", hue="Cell Type", palette=palette, s=point_size, linewidth=0, ax=ax, legend=False
        )
        ax.set_title(method)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

        # annotate scores (top-right)
        row = scores_df[scores_df["Segmentation Method"] == method]
        if not row.empty:
            (n_cells, perc_unassigned, rmsd, sil, ari, pur) = (
                row["n_cells"].iloc[0],
                row["perc_unassigned"].iloc[0],
                row["rmsd"].iloc[0],
                row["silhouette"].iloc[0],
                row["ari"].iloc[0],
                row["purity"].iloc[0],
            )
            txt = (
                f"# Cells: {n_cells}\n"
                f"Perc. unass.: {perc_unassigned:.3f}\n"
                f"RMSD: {rmsd:.3f}\n"
                f"Silhouette: {sil:.3f}\n"
                f"ARI: {ari:.3f}\n"
                f"Purity: {pur:.3f}"
            )

            ax.text(
                0.99,
                0.99,
                txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.8"),
            )

    # hide unused axes
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    # optional global legend
    if show_legend:
        handles = [Patch(color=palette[ct], label=ct) for ct in all_celltypes]
        fig.legend(handles=handles, title="Cell Type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    fig.suptitle("UMAP colored by transferred cell type", y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path / filename, bbox_inches="tight")
    plt.close(fig)


def plot_box_by_celltype_combined(
    method_to_adata: dict[str, ad.AnnData],
    celltype_col: str,
    value_key: str,
    method_palette: Mapping[str, str],
    output_path: Path,
    filename: str,
    x_order: list[str] | None = None,
    title: str | None = None,
    dropna: bool = True,
    missing_label: str = "None",
) -> pd.DataFrame:
    """
    Side-by-side boxplots for any numeric obs key (e.g., 'F1_purity', 'cell_size', ...).
    Expects box_df from build_obs_box_df.
    Uses: x="Cell Type", y="value", hue="Segmentation Method".
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    box_df = build_obs_box_df(method_to_adata, celltype_col, value_key, dropna, missing_label)

    if box_df.empty:
        raise ValueError("box_df is empty.")

    if x_order is None:
        x_order = sorted(box_df["Cell Type"].unique().tolist())

    value_key = box_df["variable"].iloc[0] if "variable" in box_df.columns else "value"

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
    fig.savefig(output_path / filename, bbox_inches="tight")
    plt.close(fig)

    return box_df


def plot_box_by_celltype(
    method_to_adata: dict[str, ad.AnnData],
    celltype_col: str,
    value_key: str,
    method_palette: Mapping[str, str],
    output_path: Path,
    filename: str,
    x_order: list[str] | None = None,
    title: str | None = None,
    dropna: bool = True,
    missing_label: str = "None",
) -> pd.DataFrame:
    """
    Stacked boxplots per method (vertical), with cell types on the x-axis.
    Only the lowest panel shows x tick labels. Y-axes are independent.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

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
    fig.savefig(output_path / filename, bbox_inches="tight")
    plt.close(fig)
    return box_df


def plot_mecr_boxplot(
    mecr_df: pd.DataFrame,
    *,
    method_palette: Mapping[str, str] | None = None,
    output_path: Path,
    filename: str = "mecr_boxplot.pdf",
) -> pd.DataFrame:
    """
    Boxplot of MECR values across any number of methods.

    Expects mecr_df from build_mecr_df with columns ['Segmentation Method','MECR'].
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if mecr_df.empty:
        raise ValueError("mecr_df is empty.")

    fig, ax = plt.subplots(figsize=(max(3.5, 1.2 * mecr_df["Segmentation Method"].nunique()), 6))
    sns.boxplot(data=mecr_df, x="Segmentation Method", y="MECR", palette=method_palette, ax=ax)
    ax.set_title("Mutually Exclusive Co-expression Rate (MECR)")
    ax.set_xlabel("Segmentation Method")
    ax.set_ylabel("MECR")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path / filename, bbox_inches="tight")
    plt.close(fig)
    return mecr_df


def save_mask_to_tiff(
    sdata,
    labels_keys: str | Sequence[str],
    output_dir: str | Path,
    scale: str = "scale0",
    tables_key: str = "table",
    obs_label_col: str = "label_id",
    obs_celltype_col: str = "transferred_celltype",
    palette: dict[str, str] | None = None,
    default_hex: str = "#808080",
    save_rgb_masks: bool = True,
    save_label_id_masks: bool = True,
    include_transcripts: bool = True,
    transcripts_key: str = "transcripts",
    x_col: str = "x",
    y_col: str = "y",
    obs_cell_id_col: str = "cell_id",
    unassigned_cell_id: str = "UNASSIGNED",
) -> None:
    """
    Save TIFF masks from SpatialData.labels. Supports one or multiple label keys.

    - If `labels_keys` is a string → one pair (RGB + label-ID TIFF) is written.
    - If `labels_keys` is a list/tuple → a pair is written per key.
    - RGB colors come from `palette` applied to `obs[obs_celltype_col]` per `obs[obs_label_col]`.
    - When `include_transcripts=True`, also rasterizes transcripts into RGB and label-ID images
      (using image size from the first labels key).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(labels_keys, str):
        labels_keys = [labels_keys]

    obs = sdata.tables[tables_key].obs

    label_to_ct = obs.set_index(obs_label_col)[obs_celltype_col]

    max_id = obs[obs_label_col].max()

    mapped = label_to_ct.map(palette)
    mapped = mapped.cat.add_categories([default_hex])
    hex_series = mapped.fillna(default_hex)

    lut = np.zeros((max_id + 1, 3), np.uint8)
    for lid, hexcol in hex_series.dropna().items():
        lut[int(lid)] = (np.array(to_rgb(hexcol)) * 255).astype(np.uint8)

    for key in labels_keys:
        da = sdata.labels[key][scale]["image"]
        labels = da.to_numpy()
        H, W = labels.shape

        if save_rgb_masks:
            rgb = lut[labels]
            tiff.imwrite(output_dir / f"{key}_{obs_celltype_col}.tif", rgb, photometric="rgb")

        if save_label_id_masks:
            tiff.imwrite(output_dir / f"{key}_{obs_label_col}.tif", labels.astype(np.uint32))

    if include_transcripts:
        pts = sdata.points[transcripts_key]
        df = pts.compute() if hasattr(pts, "compute") else pts

        id_map = dict(zip(obs[obs_cell_id_col], obs[obs_label_col], strict=False))
        id_map.setdefault(unassigned_cell_id, 0)

        x = df[x_col].astype(int).to_numpy()
        y = df[y_col].astype(int).to_numpy()
        lid = df[obs_cell_id_col].map(id_map).astype(int).to_numpy()

        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (lid >= 0) & (lid <= max_id)

        img_rgb = np.zeros((H, W, 3), np.uint8)
        img_rgb[y[ok], x[ok]] = lut[lid[ok]]
        tiff.imwrite(output_dir / f"{transcripts_key}_{obs_celltype_col}.tif", img_rgb, photometric="rgb")

        img_id = np.zeros((H, W), np.uint32)
        img_id[y[ok], x[ok]] = lid[ok]
        tiff.imwrite(output_dir / f"{transcripts_key}_{obs_label_col}.tif", img_id)
