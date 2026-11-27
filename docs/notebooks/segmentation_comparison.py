# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Comparing segmentation methods
#
# Different segmentation methods assign the same transcripts to cells in different ways. SegTraQ
# computes the same metrics for every method, so that the methods can be compared directly.
#
# We compare two segmentations of a Xenium human breast cancer sample (GEO GSM9509160): the
# default Xenium segmentation (nucleus expansion) and Proseg. Both objects contain the same
# transcripts, DAPI image and nuclei. We use a 450 µm crop, about 2,000 cells.
#
# This analysis is described in the SegTraQ manuscript (Fig. 4), where four methods are compared on
# the whole sample.

# %%
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spatialdata_plot  # noqa: F401  (adds the .pl accessor to SpatialData)

import segtraq

warnings.filterwarnings("ignore")
segtraq.settings.n_jobs = 8  # number of CPU cores

# %% [markdown]
# ## Load the data

# %%
sdata = {
    "Xenium": segtraq.datasets.xenium_breast(),
    "Proseg": segtraq.datasets.proseg_breast(),
}
sdata["Proseg"]

# %% [markdown]
# ## Run SegTraQ
#
# The Xenium object uses the SegTraQ defaults. Proseg names cells and genes differently, so we pass
# the keys. Proseg also assigns every transcript a cell ID column, with missing values for
# transcripts outside cells (`points_background_id=None`).

# %%
st = {
    "Xenium": segtraq.SegTraQ(sdata["Xenium"]),
    "Proseg": segtraq.SegTraQ(
        sdata["Proseg"],
        tables_cell_id_key="cell",
        tables_area_key=None,
        tables_centroid_x_key="centroid_x",
        tables_centroid_y_key="centroid_y",
        points_cell_id_key="assignment",
        points_gene_key="gene",
        points_background_id=None,
        shapes_cell_id_key="cell",
        nucleus_shapes_cell_id_key="cell_id",
    ),
}

# %% [markdown]
# For the supervised metrics we use a breast cancer scRNA-seq reference (Janesick et al.),
# subsampled to at most 500 cells per cell type. The slow RCTD contamination metric is off by
# default (see the [supervised module](supervised.ipynb)).

# %%
adata_ref = segtraq.datasets.reference("breast")

for method, s in st.items():
    print(method)
    s.run_all(adata_ref=adata_ref, ref_cell_type="celltype_major")

# %% [markdown]
# ## Cells coloured by cell type
#
# SegTraQ transfers cell types from the reference to the cells (`transferred_cell_type`). Cells with
# fewer than 10 transcripts or 5 genes are not labelled (grey). We plot the same 150 µm window as in
# Fig. 4a.

# %%
PX_PER_UM = 1 / 0.2125  # the coordinate system is in DAPI pixels
x0, y0, size = 2330.125 * PX_PER_UM, 1755.0625 * PX_PER_UM, 150 * PX_PER_UM

cell_types = sorted(set().union(*(s.sdata.tables["table"].obs["transferred_cell_type"].dropna() for s in st.values())))
colours = [mpl.colors.to_hex(c) for c in plt.get_cmap("tab20").colors[: len(cell_types)]] + ["#d9d9d9"]
palette = dict(zip([*cell_types, "unlabelled"], colours, strict=True))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, (method, s) in zip(axes, st.items(), strict=True):
    window = s.sdata.query.bounding_box(
        axes=("x", "y"),
        min_coordinate=[x0, y0],
        max_coordinate=[x0 + size, y0 + size],
        target_coordinate_system="global",
    )
    table = window.tables["table"]
    labels = table.obs["transferred_cell_type"].astype(object).fillna("unlabelled")
    present = [c for c in palette if c in set(labels)]  # spatialdata-plot expects one colour per category present
    table.obs["cell_type"] = pd.Categorical(labels, categories=present)
    table.uns["cell_type_colors"] = [palette[c] for c in present]
    (
        window.pl.render_images("morphology_focus", cmap="gray")
        .pl.render_shapes(
            "cell_boundaries",
            color="cell_type",
            fill_alpha=0.5,
            outline_alpha=1,
            outline_width=0.4,
            outline_color="black",
        )
        .pl.show(ax=ax, title=method, legend_loc=None if method == "Xenium" else "right margin", colorbar=False)
    )
    ax.set(xlim=(x0, x0 + size), ylim=(y0 + size, y0), xticks=[], yticks=[], xlabel="", ylabel="")
plt.tight_layout()
plt.show()

# %% [markdown]
# `segtraq.pl.celltype_proportions` compares the cell-type composition of the whole crop between the
# methods.

# %%
segtraq.pl.celltype_proportions(
    st,
    celltype_col="transferred_cell_type",
    ct_palette=palette,
    missing_label="unlabelled",
)

# %% [markdown]
# Proseg cells follow the transcripts and are not limited to a fixed distance around the nucleus.
# Cells are therefore more varied in shape.
#
# ## Compare cells with the same number of transcripts
#
# Many metrics depend on the number of transcripts per cell: with more transcripts, more
# transcripts of a neighbouring cell can be mixed in. A method that makes larger cells can look
# worse only because its cells contain more transcripts. We therefore compare cells with similar
# numbers of transcripts.
#
# We pool the cells of both methods and split them into eight bins with the same number of cells
# (octiles of transcripts per cell). Within each bin we compute the mean of each metric per method.
# Bins with fewer than 30 cells with a value are not shown.

# %%
METRICS = {  # column in table.obs: (label, +1 if higher is better, -1 if lower is better)
    "marker_balanced_accuracy": ("Marker balanced accuracy", +1),
    "contamination_strength": ("Contamination strength", -1),
    "border_admixture_score": ("Border admixture", -1),
}
cells = pd.concat(
    [s.sdata.tables["table"].obs[["transcript_count", *METRICS]].assign(method=m) for m, s in st.items()],
    ignore_index=True,
)
edges = np.unique(np.quantile(cells["transcript_count"], np.linspace(0, 1, 9)))
cells["bin"] = pd.cut(cells["transcript_count"], edges, include_lowest=True)
bin_centre = cells.groupby("bin", observed=True)["transcript_count"].median()

fig, axes = plt.subplots(1, len(METRICS), figsize=(3.2 * len(METRICS), 3), sharex=True)
for ax, (column, (label, direction)) in zip(axes, METRICS.items(), strict=True):
    for method, df in cells.groupby("method"):
        stats = df.groupby("bin", observed=True)[column].agg(["mean", "sem", "count"])
        stats = stats[stats["count"] >= 30]
        x = bin_centre.loc[stats.index]
        ax.plot(x, stats["mean"], marker="o", ms=3, label=method)
        ax.fill_between(x, stats["mean"] - 1.96 * stats["sem"], stats["mean"] + 1.96 * stats["sem"], alpha=0.2)
    ax.set(xscale="log", xlabel="Transcripts per cell", title=f"{label} {'↑' if direction > 0 else '↓'}")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ↑ higher is better, ↓ lower is better. Bands show 95 % confidence intervals.
#
# ## Summary table
#
# Finally, we summarise each method with one value per metric: the mean over cells for per-cell
# metrics, and the stored value for sample-level metrics. The better value of each metric is shown
# in bold.

# %%
SUMMARY = {  # label: (where, key, aggregation, +1 / -1 / 0 for no preferred direction)
    "Unassigned transcripts (%)": ("uns", "perc_unassigned_transcripts", None, -1),
    "Transcripts per cell": ("obs", "transcript_count", "mean", +1),
    "Genes per cell": ("obs", "gene_count", "mean", +1),
    "Cells": ("uns", "num_cells", None, 0),
    "Nucleus fraction": ("obs", "nucleus_fraction", "mean", +1),
    "IoU with nucleus": ("obs", "iou", "mean", +1),
    "Cell area (µm², median)": ("obs", "cell_area", "median", 0),
    "Marker balanced accuracy": ("obs", "marker_balanced_accuracy", "mean", +1),
    "Contamination strength": ("obs", "contamination_strength", "mean", -1),
    "Border admixture": ("obs", "border_admixture_score", "mean", -1),
    "Leiden ARI": ("uns", "mean_ari_leiden", None, +1),
    "Leiden purity": ("uns", "mean_purity_leiden", None, +1),
}


def summary_value(table, where, key, aggregation):
    if where == "uns":
        return float(np.asarray(table.uns[key])) if key in table.uns else np.nan
    return table.obs[key].agg(aggregation) if key in table.obs else np.nan


summary = pd.DataFrame(
    {
        method: {label: summary_value(s.sdata.tables["table"], *spec[:3]) for label, spec in SUMMARY.items()}
        for method, s in st.items()
    }
)


def bold_best(row):
    direction = SUMMARY[row.name][3]
    if direction == 0 or row.isna().all():
        return [""] * len(row)
    best = row.max() if direction > 0 else row.min()
    return ["font-weight: bold" if v == best else "" for v in row]


summary.style.apply(bold_best, axis=1).format("{:.3g}")

# %% [markdown]
# In this crop:
#
# - Xenium assigns almost all transcripts to cells; Proseg leaves about 6 % outside cells. Cells of
#   both methods contain a similar number of transcripts.
# - Proseg cells overlap less with the nuclei (nucleus fraction, IoU), because Proseg does not
#   grow cells from the nuclei by a fixed distance.
# - Proseg cells are purer: higher marker balanced accuracy, lower contamination strength and lower
#   border admixture, also at the same number of transcripts per cell.
# - Clustering stability (Leiden ARI and purity) is similar, slightly higher for Xenium.
#
# Which method is preferable depends on the downstream analysis. For example, cell typing needs pure
# cells, while some analyses benefit from assigning as many transcripts as possible.
#
# Cells at the edge of the crop are cut and contain fewer transcripts. For a final comparison, use
# the whole sample, as in the manuscript.
