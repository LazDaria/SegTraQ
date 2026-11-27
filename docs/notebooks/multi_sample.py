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
# # Finding low-quality samples
#
# Large studies contain many samples, and some of them may be of poor quality. SegTraQ
# summarises each sample with a few numbers, so that samples can be compared side by side.
#
# We use a CosMx tissue microarray of lung adenocarcinoma (GEO GSM9046088): 18 cores imaged as
# 36 fields of view (FOVs). Each FOV is one SpatialData object.
#
# This tutorial has two parts:
#
# 1. Run SegTraQ on one FOV. This is what you would do for each of your samples.
# 2. Compare all 36 FOVs. Here we download the objects with all SegTraQ metrics already computed.
#    We first remove FOVs with too few transcripts per cell, and then rank the remaining FOVs by
#    how unusual their metrics are.
#
# This analysis is described in the SegTraQ manuscript (Fig. 3).

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import segtraq

warnings.filterwarnings("ignore")
segtraq.settings.n_jobs = 8  # number of CPU cores

# %% [markdown]
# ## 1. Run SegTraQ on one FOV
#
# We load FOV 11. The object contains the transcripts, the cell boundaries and the cell x gene table.

# %%
sdata = segtraq.datasets.cosmx_lung_fov()
sdata

# %% [markdown]
# SegTraQ uses Xenium names by default. For CosMx, we tell it where to find the cell IDs, genes and
# z positions. There are no nucleus boundaries and no quality values, so we switch these off.
# Control probes (for example `NegPrb`) are removed by default.

# %%
st = segtraq.SegTraQ(
    sdata,
    images_key=None,
    tables_cell_id_key="cell_ID",
    tables_area_key=None,
    tables_centroid_x_key=None,
    tables_centroid_y_key=None,
    points_cell_id_key="cell_ID",
    points_gene_key="target",
    points_z_key="z_raw",
    points_background_id=0,  # transcripts with cell ID 0 are not assigned to a cell
    shapes_cell_id_key="cellID",
    nucleus_shapes_key=None,
    filter_kwargs={"min_qv": None},  # CosMx has no quality values
)

# %% [markdown]
# The supervised metrics need a single-cell RNA-seq reference with cell types. We use a lung
# adenocarcinoma reference (Lee et al.), subsampled to at most 500 cells per cell type.

# %%
adata_ref = segtraq.datasets.reference("lung")
adata_ref.obs["author_cell_type_level_1"].value_counts().head()

# %% [markdown]
# `run_all` runs every module that applies to the data. Modules that need missing inputs, such as
# nucleus boundaries, are skipped with a message.
#
# The RCTD contamination metric is slow and therefore off by default. To compute it, pass
# `supervised_kwargs={"run_rctd": True}`; see the [supervised module](supervised.ipynb). The
# precomputed objects in part 2 include it.

# %%
st.run_all(
    adata_ref=adata_ref,
    ref_cell_type="author_cell_type_level_1",
    ref_gene_key="gene_symbol",
)

# %% [markdown]
# The results are stored in the table of the SpatialData object: one value per cell in `obs`, and
# one value per sample in `uns`.

# %%
table = st.sdata.tables["table"]
table.obs[["transcript_count", "gene_count", "transferred_cell_type", "marker_balanced_accuracy"]].head()

# %%
for k in ["num_cells", "num_transcripts", "perc_unassigned_transcripts", "mean_ari_leiden"]:
    value = float(table.uns[k])
    print(f"{k}: {value:,.0f}" if value.is_integer() else f"{k}: {value:.3g}")

# %% [markdown]
# Save the object to keep the metrics, for example with `st.sdata.write("fov11_segtraq.zarr")`.

# %% [markdown]
# ## 2. Compare all FOVs
#
# We now download all 36 FOVs with the SegTraQ metrics already computed (about 670 MB).

# %%
fovs = segtraq.datasets.cosmx_lung_tma()
print(f"{len(fovs)} FOVs")

# %% [markdown]
# We summarise each FOV with one number per metric. Sample-level metrics come from `uns`. For
# per-cell metrics we take the median (or the mean) over cells.


# %%
def summarize(sdata, p_threshold=0.05):
    obs, uns = sdata.tables["table"].obs, sdata.tables["table"].uns
    row = {
        k: float(np.asarray(uns[k])) if k in uns else np.nan
        for k in [
            "num_cells",
            "num_transcripts",
            "perc_unassigned_transcripts",
            "mean_ari_leiden",
            "mean_purity_leiden",
            "silhouette_score_transferred_cell_type",
            "connectedness_transferred_cell_type",
        ]
    }
    for k in ["transcript_count", "gene_count", "marker_balanced_accuracy", "rctd_contamination"]:
        row[f"{k}_med"] = obs[k].median() if k in obs else np.nan  # missing if the module was skipped
    row["contamination_strength_mean"] = obs["contamination_strength"].mean()
    # border admixture: mean over cells, counting non-significant or negative scores as 0
    p, score = obs["border_admixture_p_value"], obs["border_admixture_score"]
    valid = p.notna() & score.notna()
    row["border_admixture_sig_mean"] = np.where((p < p_threshold) & (score > 0), score, 0)[valid].mean()
    return row


metrics = pd.DataFrame({fov: summarize(s) for fov, s in fovs.items()}).T
metrics.round(2).head()

# %% [markdown]
# ### Remove FOVs with too few transcripts per cell
#
# Cells with few transcripts make most metrics unreliable. We remove FOVs with a median below 100
# transcripts per cell. The plot shows, for each FOV, the fraction of cells (y) with at most a given
# number of transcripts (x). Removed FOVs are shown in red.

# %%
MIN_MEDIAN_TRANSCRIPTS = 100
low_coverage = metrics.index[metrics["transcript_count_med"] < MIN_MEDIAN_TRANSCRIPTS].tolist()

fig, ax = plt.subplots(figsize=(4, 3.5))
for fov, s in fovs.items():
    x = np.sort(s.tables["table"].obs["transcript_count"].to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)
    low = fov in low_coverage
    ax.plot(x[x > 0], y[x > 0], color="#c93e3f" if low else "#c8ceda", lw=1 if low else 0.7, zorder=3 if low else 2)
ax.axvline(MIN_MEDIAN_TRANSCRIPTS, color="0.3", ls=":", lw=1)
ax.set(xscale="log", xlabel="Transcripts per cell", ylabel="Fraction of cells", ylim=(0, 1.01))
ax.set_title(f"red: median below {MIN_MEDIAN_TRANSCRIPTS} transcripts per cell ({len(low_coverage)} FOVs)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.show()
print("removed FOVs:", low_coverage)

# %% [markdown]
# ### Rank the remaining FOVs
#
# `segtraq.sample_anomaly_scores` compares each FOV with the median of all FOVs:
#
# - Each metric becomes a robust z-score. It is multiplied by the direction of the metric, so that
#   positive values are always better than the median.
# - Values better than the median are set to 0. Only deviations towards poor quality count.
# - An isolation forest scores how isolated each FOV is from the others. A high score means that
#   the FOV is unusual.
#
# The isolation forest assumes that most samples are typical, so it needs many samples (here 27).
#
# For each metric we give its direction: +1 if higher is better, -1 if lower is better.

# %%
DIRECTIONS = {
    "num_cells": +1,
    "num_transcripts": +1,
    "transcript_count_med": +1,
    "gene_count_med": +1,
    "perc_unassigned_transcripts": -1,
    "border_admixture_sig_mean": -1,
    "contamination_strength_mean": -1,
    "rctd_contamination_med": -1,
    "marker_balanced_accuracy_med": +1,
    "mean_ari_leiden": +1,
    "mean_purity_leiden": +1,
    "silhouette_score_transferred_cell_type": +1,
    "connectedness_transferred_cell_type": +1,
}

kept = metrics.drop(index=low_coverage)
z, anomaly = segtraq.sample_anomaly_scores(kept, DIRECTIONS)
anomaly.sort_values(ascending=False).head().round(3)

# %% [markdown]
# The heatmap shows the z-scores of each FOV (columns), sorted by the anomaly score (bars). Pink
# means worse than the median, green better.

# %%
order = anomaly.sort_values(ascending=False).index
fig, (ax_bar, ax_heat) = plt.subplots(
    2, 1, figsize=(9, 4.5), sharex=True, gridspec_kw={"height_ratios": [1, 4], "hspace": 0.05}
)
ax_bar.bar(range(len(order)), anomaly[order], color="#99a3b5")
ax_bar.set_ylabel("Anomaly\nscore")
ax_bar.spines[["top", "right"]].set_visible(False)
im = ax_heat.imshow(z.loc[order].T.to_numpy(float), aspect="auto", cmap="PiYG", vmin=-2.5, vmax=2.5)
ax_heat.set_xticks(range(len(order)), order)
ax_heat.set_yticks(range(len(DIRECTIONS)), list(DIRECTIONS))
ax_heat.set_xlabel("FOV")
fig.colorbar(im, ax=[ax_bar, ax_heat], shrink=0.6, label="z-score (+ = better)")
plt.show()

# %% [markdown]
# The ranking is the same as in the manuscript (Fig. 3d):
#
# - FOV 16 has the highest score. It has few transcripts and genes per cell, just above the
#   threshold, and its cell types form poorly connected clusters (cluster connectedness).
# - FOVs 43 and 44 follow. Their clusters are less stable (ARI, purity), and in FOV 43 cells
#   contain more transcripts of other cell types (contamination strength).
# - FOV 5 has many unassigned transcripts.
# - FOVs 26 and 11 have the lowest scores. Their metrics are close to the median.
#
# These FOVs are worth a closer look before they are used in downstream analyses.
