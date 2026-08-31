# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: segtraq_26
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Technology Focus: 10x Genomics Xenium
#
# To follow along with this tutorial, you can download the data already in
# `SpatialData` format from [here](https://oc.embl.de/index.php/s/YSvZTt8AArh4c5a).

# %% [markdown]
# ## Description of prior dataset acquisition and segmentation
#
# ### Dataset Acquisition (10x Genomics Xenium)
# For this tutorial, we used data from the **10x Genomics Xenium Prime Breast Cancer FFPE** dataset.
# We downloaded the data using:
# ```bash
# curl -O https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/3.0.0/
# Xenium_Prime_Breast_Cancer_FFPE/Xenium_Prime_Breast_Cancer_FFPE_outs.zip
# ```
#
# ### Subsetting the Field of View
# We subset the dataset using the following **bounding box** (X, Y coordinates):
# ```
# 2631.31, 8503.97
# 4150.72, 8503.97
# 4150.72, 10298.85
# 2631.31, 10298.85
# ```
#
# ### Cell Segmentation Methods
# We applied two segmentation approaches for comparison to the
# [Xenium Multi-Modal Cell Segmentation](https://www.10xgenomics.com/support/software/
# xenium-onboard-analysis/3.1/algorithms-overview/segmentation):
# 1. **BIDCell** — [https://github.com/SydneyBioX/BIDCell](https://github.com/SydneyBioX/BIDCell)
# 2. **Proseg** (v2.0.5) — [https://github.com/dcjones/proseg](https://github.com/dcjones/proseg)
#
# ### Reference scRNA-seq Dataset for Training & QC
# For **BIDCell** training, as well as supervised quality control (see below in this tutorial),
# we used the **Janesick et al.** scRNA-seq breast cancer dataset following:
# * LMWeber Script on loading Chromium data:
#   [https://github.com/lmweber/STexampleData/blob/devel/inst/scripts/
# make-data-Janesick-breastCancer-Chromium.R](https://github.com/lmweber/STexampleData/blob/devel/inst/
# scripts/make-data-Janesick-breastCancer-Chromium.R)
# * OSTA Guide on processing the Chromium data:
#   [https://lmweber.org/OSTA/pages/seq-deconvolution.html](https://lmweber.org/OSTA/pages/
# seq-deconvolution.html#:~:text=e.%C2%A0Annogrp\).\-,Code\,-%23%20retrieve%20dataset%20from)
#
# Download commands:
# ```bash
# wget -O count_matrix.h5 "https://www.ncbi.nlm.nih.gov/geo/download/
# ?acc=GSM7782698&format=file&file=GSM7782698%5Fcount%5Fraw%5Ffeature%5Fbc%5Fmatrix%2Eh5"
# wget -O Cell_Barcode_Type_Matrices.xlsx "https://zenodo.org/records/10076046/files/
# Cell_Barcode_Type_Matrices.xlsx?download=1"
# ```
#
# ### Segmentation Parameters
#
# - Proseg
#     * Used **default parameters**.
#
# - BIDCell
#     * Annotated elongated cell types:
#       * endo
#       * stromal
#       * myoepi
#       * perivas
#
#     * Additional settings:
#       * `test_epoch = 4`
#       * `test_steps = 100`
#
#     * All other parameters left at default.
#
# ### Import into SpatialData
# After segmentation, all outputs were imported into **SpatialData** objects.
# For a demonstration on how to get data into the right format,
# please refer to this [tutorial](io.ipynb).
# These processed SpatialData objects are also uploaded to the
# [tutorial repository](https://oc.embl.de/index.php/s/1JyN4Qvk4mw0T5J)
# so users can directly run the workflow without repeating the preprocessing steps.

# %% [markdown]
# ## Read and crop SpatialData objects

# %%

import anndata as ad
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import spatialdata as sd
import spatialdata_plot  # noqa

import segtraq

segtraq.settings.n_jobs = -1  # Use all available CPU cores

# warnings.filterwarnings(action="ignore")

# %%
sdata_xenium_ws = sd.read_zarr("../../data/xenium_5K_data/xenium.zarr")
sdata_bidcell_ws = sd.read_zarr("../../data/xenium_5K_data/bidcell.zarr")
sdata_proseg_ws = sd.read_zarr("../../data/xenium_5K_data/proseg2.zarr")

# %% [markdown]
# The `SpatialData` objects can be subset via `spatialdata.query.bounding_box()`.

# %%
bb_xmin = 800
bb_ymin = 1150
bb_w = 200
bb_h = 300
bb_xmax = bb_xmin + bb_w
bb_ymax = bb_ymin + bb_h

# %%
f, ax = plt.subplots(figsize=(5, 5))
sdata_xenium_ws.pl.render_shapes("cell_boundaries").pl.show(ax=ax)
rect = patches.Rectangle((bb_xmin, bb_ymin), bb_w, bb_h, linewidth=5, edgecolor="red", facecolor="none")
ax.add_patch(rect)


# %%
def crop(ws):
    return ws.query.bounding_box(
        axes=["x", "y"],
        min_coordinate=[bb_xmin, bb_ymin],
        max_coordinate=[bb_xmax, bb_ymax],
        target_coordinate_system="global",
    )


sdata_xenium = crop(sdata_xenium_ws)
sdata_bidcell = crop(sdata_bidcell_ws)
sdata_proseg = crop(sdata_proseg_ws)

# %% [markdown]
# ## Initialize SegTraQ objects
#
# Next, we initialize SegTraQ objects, the core interface for computing SegTraQ metrics.
# During initialization, all inputs are are validated via `validate_spatialdata()`.
# Don't worry if you do not know which parameters you need up front;
# just put in your spatialdata object, and SegTraQ will tell you which arguments are wrong/missing.
#
# The four segmentation methods apply different filters to the points prior to segmentation.
# Proseg and BIDCell remove control probes (e.g. NegControlProbe, Intergenic_Region)
# before segmentation. In addition, Proseg and BIDCell exclude
# transcripts with a quality value (qv, the Xenium per-transcript decoding quality score) < 20.
#
# For a fair comparison of metrics, like the percentage of unassigned transcripts,
# across methods, we subset the transcripts to include only gene-specific probes with qv ≥ 20.
# This is also compatible with the preprocessing for the cell feature matrix by
# [10x Genomics](https://cf.10xgenomics.com/supp/xenium/xenium_documentation.html).

# %%
st_xenium = segtraq.SegTraQ(
    sdata_xenium, images_key="image", tables_centroid_x_key="x_centroid", tables_centroid_y_key="y_centroid"
)
st_bidcell = segtraq.SegTraQ(
    sdata_bidcell,
    images_key="image",
    points_background_id=0,
    tables_area_key=None,
    tables_centroid_x_key="cell_centroid_x",
    tables_centroid_y_key="cell_centroid_y",
)
st_proseg = segtraq.SegTraQ(
    sdata_proseg,
    images_key="image",
    points_cell_id_key="assignment",
    points_background_id=2**32 - 1,
    points_gene_key="gene",
    tables_area_key=None,
    tables_cell_id_key="cell",
    shapes_cell_id_key="cell",
    tables_centroid_x_key="centroid_x",
    tables_centroid_y_key="centroid_y",
)

# %% [markdown]
# ## Load reference scRNA-seq data and transfer labels

# %% [markdown]
# To evaluate how SegTraQ metrics differ between cell types,
# we first transfer labels from the reference scRNA-seq dataset to the spatial data using
# `segtraq.run_label_transfer`.

# %%
# Load scRNA-seq data
adata_ref = ad.read_h5ad("../../data/xenium_5K_data/BC_scRNAseq_Janesick.h5ad")

# %% [markdown]
# For easier access, we store the `SpatialData` object into a dictionary.

# %%
st_dict = {"xenium": st_xenium, "bidcell": st_bidcell, "proseg": st_proseg}

# %%
# Transfer labels
for _method, st in st_dict.items():
    st.run_label_transfer(adata_ref, ref_cell_type="celltype_major", ref_raw_counts_layer="raw", inplace=True)

# %% [markdown]
# We can visualize the spatial data and color cell masks by transferred labels using `spatialdata-plot`.

# %%
# Define color palette for plotting
col_celltype = {
    "T": "#fb8072",
    "B": "#bc80bd",
    "macro": "#910290",
    "dendritic": "#fdb462",
    "mast": "#959059",
    "perivas": "#fed9a6",
    "endo": "#a6cee3",
    "myoepi": "#2782bb",
    "DCIS1": "#3c7761",
    "DCIS2": "#66a61e",
    "tumor": "#66c2a5",
    "stromal": "#d45943",
    "Unknown": "#808080",
}

# %% [markdown]
# Cells with a transcripts count < 10 and a gene count < 5 are not considered in
# label transfer and their returned `transferred_cell_type` returns `np.nan`,
# and missing values are subsequently replaced with the label "Unknown".

# %%
# Replace NaN with Unknown for plotting
for _method, st in st_dict.items():
    s = st.sdata.tables["table"].obs["transferred_cell_type"]
    if pd.api.types.is_categorical_dtype(s):
        s = s.cat.add_categories(["Unknown"])

    st.sdata.tables["table"].obs["transferred_celltype_plot"] = s.fillna("Unknown")

# %%
plt.style.use("dark_background")


# %%
def feature_spatial_plot(sdata, shapes_key, feature, col_palette, method):
    axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)[1].flatten()

    # compute per-cell color
    labels = sdata.tables["table"].obs[feature].unique().astype(str).tolist()
    cols = [col_palette[lab] for lab in labels]

    # Dapi image
    sdata.pl.render_images("image").pl.show(ax=axes[0], title=f"{method}: DAPI image", coordinate_systems="global")

    # Plot - Cell boundaries
    sdata.tables["table"].obs["region"] = shapes_key
    sdata.set_table_annotates_spatialelement("table", region=shapes_key)

    sdata.pl.render_shapes(shapes_key, color=feature, palette=cols, groups=labels).pl.show(
        ax=axes[1], title=f"{method}: Cell masks colored by {feature}", coordinate_systems="global"
    )


# %% [markdown]
# *10 Xenium MultiModal Cell Segmentation*

# %%
feature_spatial_plot(st_xenium.sdata, "cell_boundaries", "transferred_celltype_plot", col_celltype, "xenium")

# %% [markdown]
# *BIDCell Segmentation*

# %%
feature_spatial_plot(st_bidcell.sdata, "cell_boundaries", "transferred_celltype_plot", col_celltype, "BIDCell")

# %% [markdown]
# *Proseg Segmentation*

# %%
feature_spatial_plot(st_proseg.sdata, "cell_boundaries_z1", "transferred_celltype_plot", col_celltype, "Proseg")

# %% [markdown]
# The dominant cell types are DCIS2, T and B cells. Proseg has the highest
# proportion of cells assigned to "Unknown", i.e. transcript counts <10 and gene_counts <5.

# %%
method_dfs = []
for method, st in st_dict.items():
    df = st.sdata["table"].obs["transferred_celltype_plot"].value_counts(normalize=True).reset_index()
    df.columns = ["celltype", "proportion"]
    df["method"] = method
    method_dfs.append(df)

plot_df = pd.concat(method_dfs, ignore_index=True)
plot_wide = plot_df.pivot(index="method", columns="celltype", values="proportion").fillna(0)

method_order = ["xenium", "bidcell", "proseg"]
plot_wide = plot_wide.loc[method_order]

fig, ax = plt.subplots(figsize=(2, 4))

plot_wide.plot(kind="bar", stacked=True, color=[col_celltype.get(c, "grey") for c in plot_wide.columns], ax=ax)

ax.set_ylabel("Proportion")
ax.set_xlabel("Method")

# Legend text in white
legend = ax.legend(title="Cell type", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Running SegTraQ QC metrics

# %% [markdown]
# Now the data is ready to compute SegTraQ QC metrics.
# Below we demonstrate how to run the individual modules.

# %% [markdown]
# ### Baseline module

# %% [markdown]
# The baseline (`bl`) module computes basic quality-control metrics such as the number of cells,
# the percentage of unassigned transcripts, and the number of transcripts and genes per cell.
# When `inplace=True`, global metrics are stored in the `uns` slot, and per-cell metrics
# are stored in the `obs` slot of the table within the `SpatialData` object.

# %%
for method, st in st_dict.items():
    st.bl.num_cells(inplace=True)
    st.bl.perc_unassigned_transcripts(inplace=True)
    st.bl.transcripts_per_cell()
    st.bl.genes_per_cell()
    st.bl.mean_transcripts_per_gene_per_cell()

    num_cells = st.sdata.tables["table"].uns["num_cells"]
    p_unassigned = st.sdata.tables["table"].uns["perc_unassigned_transcripts"]
    mean_tx = st.sdata.tables["table"].obs["transcript_count"].mean()
    mean_gn = st.sdata.tables["table"].obs["gene_count"].mean()
    mean_tgc = st.sdata.tables["table"].obs["mean_transcripts_per_gene"].mean()

    print(f"\n{method}")
    print(f"#cells: {num_cells}; %unassigned: {p_unassigned}")
    print(f"mean #transcripts: {mean_tx}; mean #genes: {np.mean(mean_gn)}")
    print(f"mean #transcripts per gene per cell: {mean_tgc}")

# %% [markdown]
# The number of detected cells is comparable across Xenium, BIDCell, and Proseg.
# Proseg leaves the smallest fraction of transcripts unassigned (0.81%).
# The number of transcripts and genes per cell can be visualized per cell type, as shown below.


# %%
# method for visualization
def boxplot_per_celltype(st_dict, feature, q=1):
    dfs = []
    for method, st in st_dict.items():
        obs = st.sdata["table"].obs[st.sdata["table"].obs["transferred_cell_type"].notna()]
        tmp = obs[["transferred_cell_type", feature]].copy()
        tmp["method"] = method
        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)
    df = df[(df[feature] <= df[feature].quantile(q))]

    fig, ax = plt.subplots(figsize=(15, 5))

    sns.boxplot(
        data=df,
        x="transferred_cell_type",
        y=feature,
        hue="method",
        showcaps=True,
        showfliers=False,
        palette="Set2",
        ax=ax,
    )

    fig.tight_layout()
    plt.show()


# %% [markdown]
# Proseg yields the highest mean number of transcripts per cell,
# driven by increased transcript detection in DCIS2 and T, B and stromal cells,
# which constitute the dominant cell populations in the dataset.
# In contrast, Proseg detects fewer transcripts in some of the less abundant cell types,
# such as DCIS1 and myoepithelial cells.

# %%
boxplot_per_celltype(st_dict, "transcript_count")

# %% [markdown]
# The number of genes detected per cell correlates strongly with the number of transcripts per cell.

# %%
boxplot_per_celltype(st_dict, "gene_count")

# %% [markdown]
# On average, the number of transcripts per gene per cell is approximately 1,
# indicating that across methods we detect roughly one to two transcripts per gene per cell.
# Among the methods, Proseg shows the highest mean number of transcripts per gene
# per cell when the *Unknown* cluster (cells with transcript counts < 10 and gene counts < 5) is excluded.
#
# Interestingly, Proseg detects more transcripts per gene per cell across all cell types,
# even in cases where the total transcript and gene counts per cell are lower,
# e.g., DCIS1 and myoepithelial cells.
# This pattern suggests that transcripts are concentrated among fewer genes per cell,
# potentially reflecting more homogeneous within-cell expression profiles (i.e. reduced gene-level variability).

# %%
boxplot_per_celltype(st_dict, "mean_transcripts_per_gene")

# %% [markdown]
# The baseline module also facilitates the computation of **morphological features**.

# %%
for _method, st in st_dict.items():
    st.bl.morphological_features(n_jobs=8)

# %%
features = [
    "cell_area",
    "perimeter",
    "circularity",
    "solidity",
    "convexity",
    "elongation",
    "eccentricity",
    "compactness",
]

# Collect features into one dataframe
all_feats = []
for method, st in st_dict.items():
    obs = st.sdata["table"].obs[st.sdata["table"].obs["transferred_cell_type"].notna()]
    feat = obs[features]
    tmp = feat.copy()
    tmp["method"] = method
    all_feats.append(tmp)

df = pd.concat(all_feats, ignore_index=True)
df["method"] = df["method"].astype(str)

feature_cols = [c for c in df.columns if c != "method"]

fig, ax = plt.subplots(2, 6, figsize=(6 * 4, 2 * 3))
ax = ax.flatten()

for i, feat_name in enumerate(feature_cols):
    sns.kdeplot(data=df, x=feat_name, hue="method", palette="Set2", common_norm=False, fill=False, ax=ax[i])
    ax[i].set_title(f"Distribution of {feat_name}")
    ax[i].set_xlabel(feat_name)
    ax[i].set_ylabel("Density")
fig.tight_layout()
plt.show()

# %% [markdown]
# Xenium detects rounder cells, characterized by higher circularity and solidity,
# while BIDCell recovers a wider distribution of cell shapes,
# including both low and high circularity and solidity.
# BIDCell incorporates a loss term enforcing cell type–specific elongation,
# which may explain the detection of less circular cell shapes.
# Notably, the same cell types specified as elongated during training
# (perivascular, myoepithelial, stromal, and endothelial cells) also exhibit the highest
# median elongation in the results, highlighting the strong influence of the prior on
# the inferred cell shapes.

# %%
obs = st_bidcell.sdata["table"].obs[st_bidcell.sdata["table"].obs["transferred_cell_type"].notna()]
df = obs[["transferred_cell_type", "elongation"]]
medians = df.groupby("transferred_cell_type").median()
order = medians.sort_values(by="elongation").index

fig, ax = plt.subplots(figsize=(10, 3))

sns.boxplot(
    data=df,
    x="transferred_cell_type",
    y="elongation",
    order=order,
    showcaps=True,
    showfliers=False,
    palette=col_celltype,
    ax=ax,
)

ax.set_title("BIDCell - cell shape representation")
fig.tight_layout()
plt.show()

# %% [markdown]
# Proseg, in contrast, does not rely solely on membrane stainings.
# Instead, it incorporates transcript positions,
# which may extend beyond the membrane boundary due to diffusion.
# This may explain why perivascular or myoepithelial cells do not appear
# among the most elongated cell types. Importantly, a segmentation mask that accurately
# delineates cell geometry may still misassign transcripts
# (e.g. due to diffusion or spatial overlap), while a mask that poorly represents
# cell shape may nonetheless correctly assign transcripts.
# Therefore, it is essential to evaluate not only the geometric accuracy of
# cell shapes but also the accuracy of transcript assignment to cells.
# We will compute metrics to evaluate transcript assignment accuracy below.

# %%
obs = st_proseg.sdata["table"].obs[st_proseg.sdata["table"].obs["transferred_cell_type"].notna()]
df = obs[["transferred_cell_type", "elongation"]]
medians = df.groupby("transferred_cell_type").median()
order = medians.sort_values(by="elongation").index

fig, ax = plt.subplots(figsize=(10, 3))

sns.boxplot(
    data=df,
    x="transferred_cell_type",
    y="elongation",
    order=order,
    showcaps=True,
    showfliers=False,
    palette=col_celltype,
    ax=ax,
)

ax.set_title("Proseg - cell shape representation")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Clustering stability module
# The clustering stability (`cs`) module provides metrics for assessing the stability
# of clustering results across different resolutions and random subsets of genes.

# %% [markdown]
# Let`s first perform Leiden clustering and visualize the clusters in the UMAP space.

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs = axs.flatten()

for i, (method, st) in enumerate(st_dict.items()):
    adata = st.sdata.tables["table"].copy()
    adata.layers["counts"] = adata.X
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

    sc.pl.umap(
        adata,
        color="leiden",
        palette="Set2",
        ax=axs[i],
        show=False,
        title=method,
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# Next, we compute clustering stability metrics and plot these.

# %%
ccds = {}
silhouette_scores = {}
purities = {}
aris = {}

for method, st in st_dict.items():
    ccds[method] = st.cs.cluster_connectedness()
    silhouette_scores[method] = st.cs.silhouette_score()
    purities[method] = st.cs.purity()
    aris[method] = st.cs.adjusted_rand_index()

# %%
results_df = pd.DataFrame(
    {
        "Method": list(ccds.keys()),
        "Connectedness": list(ccds.values()),
        "Silhouette Score": list(silhouette_scores.values()),
        "Purity": list(purities.values()),
        "ARI": list(aris.values()),
    }
)

# %%
fig, ax = plt.subplots(figsize=(4, 3))

sns.scatterplot(
    data=results_df,
    x="Connectedness",
    y="Silhouette Score",
    hue="Method",
    style="Method",
    s=100,
    palette="Set2",
    ax=ax,
)

ax.set_xlabel("Connectedness (↑)")
ax.set_ylabel("Silhouette Score (↑)")
ax.set_title("Connectedness vs. Silhouette Score")
ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)
legend = ax.legend(title="Method", frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")

fig.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(4, 3))

sns.scatterplot(
    data=results_df,
    x="Purity",
    y="ARI",
    hue="Method",
    style="Method",
    s=100,
    palette="Set2",
    ax=ax,
)

ax.set_xlabel("Purity (↑)")
ax.set_ylabel("ARI (↑)")
ax.set_title("Purity vs. ARI")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

legend = ax.legend(title="Method", frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")

fig.tight_layout()
plt.show()

# %% [markdown]
# Proseg achieves the highest adjusted Rand index (ARI) and clustering purity,
# as well as the lowest mean cosine distance. Together,
# these results suggest that Proseg retrieves the most internally consistent expression
# profiles and could indicate less contaminated single cell signatures.
#
# The silhouette score can also be computed on the transferred labels.
# We will first visualize transferred labels in the UMAP space.

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs = axs.flatten()

for i, (method, st) in enumerate(st_dict.items()):
    adata = st.sdata.tables["table"].copy()

    # normalizing and log-transforming the counts
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    # computing a PCA and neighbors
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    sc.pl.umap(
        adata,
        color="transferred_cell_type",
        palette=col_celltype,
        ax=axs[i],
        show=False,
        title=method,
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# Proseg achieves the highest silhouette score for the transferred labels,
# indicating higher intra-cluster coherence and improved inter-cluster separation;
# this is also visually consistent with the UMAP embedding.

# %%
silhouette_scores_labels = {}

for method, st in st_dict.items():
    silhouette_scores_labels[method] = st.cs.silhouette_score(cell_type_key="transferred_cell_type")

silhouette_scores_labels

# %% [markdown]
# ## Region similarity module
#
# While individual genes may exhibit subcellular localization patterns,
# the overall distribution of transcripts, when averaged across genes,
# is expected to be relatively smooth and approximately uniform within a cell.
# Based on this assumption, the region similarity module evaluates the
# similarity of gene expression profiles across different subcellular compartments.
# Deviations from this expected intra-cellular consistency can serve as indicators of
# transcript contamination originating from neighboring cells.

# %% [markdown]
# *Intersection over Union (IoU)*

# %% [markdown]
# Proseg shows the lowest intersection over union (IoU) between the cell and nucleus.
# It relies on transcript assignment rather than explicit morphology-based segmentation,
# allowing transcripts that are repositioned outside the cell membrane
# (e.g. due to transcript diffusion) to be reassigned to the cell of origin.
# This often results in larger inferred cell areas (or perimeters, as shown above)
# relative to the nucleus or less aligned cell and nucleus shapes, leading to
# lower cell–nucleus IoU values.

# %%
plt.style.use("default")


def plot_IoU(sdata, shapes_key, method, ax):
    # note that copy.deepcopy() can have unwanted effects on the attrs, hence sd.deepcopy() is preferred
    sdata_plot = sd.deepcopy(sdata)
    sdata_plot["table"] = sdata_plot["table"][(sdata_plot["table"].obs["iou"].notna())]

    sdata_plot.tables["table"].obs["region"] = shapes_key
    sdata_plot.set_table_annotates_spatialelement("table", region=shapes_key)

    sdata_plot.pl.render_shapes(
        element="nucleus_boundaries",
        fill_alpha=0.2,
        outline_alpha=1.0,
        outline_width=0.5,
        outline_color="black",
    ).pl.render_shapes(
        element=shapes_key,
        color="iou",
        cmap="viridis",
        fill_alpha=0.5,
        outline_alpha=1.0,
        outline_width=0.5,
        outline_color="black",
    ).pl.show(ax=ax, title=f"{method}: Overlay of nuclei and cell masks colored by IoU", colorbar=True)


# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

for i, (method, st) in enumerate(st_dict.items()):
    st.rs.match_nuclei_to_cells(n_jobs=-1)
    plot_IoU(st.sdata, "cell_boundaries", method, axes[i])

# %% [markdown]
# *Similarity between expression in cell and nucleus*
#
# Once, the "best-matching" nucleus is determined for each cell based on IoU,
# we can compute the similarity between cell and nuclear expression.

# %%
for _method, st in st_dict.items():
    st.rs.similarity_nucleus_cell(n_jobs=-1)


# %%
def histogram_feature(
    sdata_dict,
    feature,
    figsize=(8, 4),
    significant_only=True,
    bins=30,
):
    plt.style.use("default")
    all_feats = []

    for method, st in sdata_dict.items():
        obs = st.sdata[st.tables_key].obs

        if significant_only:
            p_value_col = f"{feature}_p_value"
            if feature == "border_admixture_score":
                p_value_col = "border_admixture_p_value"
            obs = obs.loc[obs[p_value_col] < 0.05]

        feat = obs[feature].to_frame(feature)
        feat["method"] = method
        all_feats.append(feat)

    df = pd.concat(all_feats, ignore_index=True)
    df["method"] = df["method"].astype(str)

    methods = list(sdata_dict.keys())
    palette = dict(zip(methods, sns.color_palette("Set2", len(methods)), strict=False))

    fig, ax = plt.subplots(figsize=figsize)

    sns.histplot(
        data=df,
        x=feature,
        hue="method",
        bins=bins,
        stat="count",
        element="step",
        # edgecolor=None,
        palette=palette,
        ax=ax,
        alpha=0.3,
        line_kws=dict(color="method", linewidth=2.5, label="KDE"),
        kde=True,
    )

    # Median for each method
    for method in methods:
        median = df.loc[df["method"] == method, feature].median()

        ax.axvline(
            median,
            color=palette[method],
            linestyle="--",
            linewidth=2,
        )

    ax.set_ylabel("Number of cells")
    plt.tight_layout()


# %% [markdown]
# Among cells with unexpectedly low nucleus–cell similarity
# (`similarity_nucleus_cell_p_value` < 0.05), ProSeg shows more negative residuals
# (`similarity_nucleus_cell`) than Xenium and BIDCell, indicating a stronger difference between the nuclear
# and whole-cell expression profiles than expected. This is partly expected from the different assignment
# strategies. Xenium and BIDCell assign transcripts to cells based on spatial overlap with the inferred cell
# boundaries, whereas ProSeg can assign transcripts to a cell even when they lie outside its final 2D cell
# boundary. Because the nuclear expression profile is also defined by spatial overlap with the nuclear boundary,
# the nucleus and whole-cell profiles are more spatially coupled for Xenium and BIDCell. Accordingly, among
# these cells, Xenium and BIDCell show less negative residuals than ProSeg. Thus, ProSeg's lower nucleus–cell
# similarity may partly reflect its transcript reassignment strategy rather than poorer segmentation.

# %%
histogram_feature(st_dict, "similarity_nucleus_cell", significant_only=True)

# %% [markdown]
# While `similarity_nucleus_cell` compares the nuclear transcript profile with the final whole-cell expression
# profile, `similarity_nucleus_cytoplasm` more directly compares two spatial regions within the cell. It compares
# the cell's transcripts regionally overlapping with the nucleus and the non-nuclear region (cytoplasm). It
# therefore asks whether the transcript composition is consistent across these two regions of the same cell.
# Here, Proseg again shows the lowest consistency between nuclear and cytoplasmic expression profiles.

# %%
for _method, st in st_dict.items():
    st.rs.similarity_nucleus_cytoplasm(n_jobs=-1)

# %%
histogram_feature(st_dict, "similarity_nucleus_cytoplasm")

# %% [markdown]
# *Admixture (contamination) of the cell border*
#
# Unlike the nucleus-based similarity metrics, the `border_admixture_score` more directly tests for contamination
# from neighboring cells. Rather than asking whether two regions within a cell have different expression profiles,
# it asks whether the cell border specifically resembles a mixture of the cell center and the surrounding
# neighborhood more strongly than expected from finite transcript sampling alone. This helps distinguish general
# within-cell expression differences from differences that specifically point toward a contribution from
# neighboring cells.
#
# Here, Xenium has the largest number of cells with unexpectedly strong border admixture
# (`border_admixture_p_value` < 0.05), meaning that for more Xenium cells, the border resembles the surrounding
# neighborhood more than would be expected by chance. Xenium also shows the largest median positive residual
# among these cells, indicating that the excess neighborhood-like signal at the border tends to be stronger.
# In contrast, BIDCell and Proseg — both transcript-informed segmentation methods — show substantially fewer
# cells with evidence of unexpected border admixture, and the effect is generally weaker when it occurs.

# %%
for _method, st in st_dict.items():
    st.rs.border_admixture_score(n_jobs=-1)

# %%
histogram_feature(st_dict, "border_admixture_score")

# %% [markdown]
# While it is important to consider metrics that do not depend on the availability of
# high-quality reference scRNA-seq data—and therefore do not rely on the performance of
# label transfer—a limitation of the region similarity module is that it cannot
# distinguish between homogeneous and heterogeneous cellular neighborhoods.
# In homogeneous neighborhoods, a high `border_admixture_score` does not necessarily
# indicate contamination. Nevertheless, when comparing methods globally or on a
# cell-type level, this is a useful unsupervised contamination metric that does
# not rely on reference scRNA-seq data.

# %% [markdown]
# When cell type labels are available, the metric can still be compared on a cell-type level. The plot compares
# `border_admixture_score` across segmentation methods within each transferred cell type, restricted to cells
# with unexpectedly strong border admixture (`border_admixture_p_value` < 0.05). The individual points and
# boxplots show the magnitude of the excess neighborhood-like signal at the border, while the annotated cell
# counts show how many cells of each cell type are significantly affected. This helps reveal whether differences
# between segmentation methods are driven by particular cell types rather than representing a global effect.


# %%
def boxplot_per_celltype_sig(st_dict, feature, q=1, significant_only=True):
    dfs = []
    for method, st in st_dict.items():
        obs = st.sdata[st.tables_key].obs[st.sdata[st.tables_key].obs["transferred_cell_type"].notna()].copy()

        if (feature == "similarity_top_bottom") and significant_only:
            obs = obs.loc[obs["similarity_top_bottom_p_value"] < 0.05].copy()

        obs["transferred_cell_type"] = obs["transferred_cell_type"].cat.remove_unused_categories()

        tmp = obs[["transferred_cell_type", feature]].copy()
        tmp["method"] = method
        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)
    df = df[df[feature] <= df[feature].quantile(q)]

    fig, ax = plt.subplots(figsize=(15, 5))

    sns.boxplot(
        data=df,
        x="transferred_cell_type",
        y=feature,
        hue="method",
        showcaps=True,
        showfliers=False,
        palette="Set2",
        boxprops={"alpha": 0.5},
        ax=ax,
    )

    sns.stripplot(
        data=df,
        x="transferred_cell_type",
        y=feature,
        hue="method",
        dodge=True,
        jitter=True,
        palette="Set2",
        alpha=1,
        size=3,
        legend=False,
        ax=ax,
    )

    celltypes = df["transferred_cell_type"].unique()
    methods = list(st_dict.keys())

    n_methods = len(methods)
    box_width = 0.8
    offsets = np.linspace(
        -box_width / 2 + box_width / (2 * n_methods),
        box_width / 2 - box_width / (2 * n_methods),
        n_methods,
    )

    # Position labels slightly above the data
    ymin, ymax = ax.get_ylim()
    y_text = ymax + 0.02 * (ymax - ymin)

    for i, celltype in enumerate(celltypes):
        for j, method in enumerate(methods):
            n = len(df.loc[(df["transferred_cell_type"] == celltype) & (df["method"] == method)])

            ax.text(
                i + offsets[j],
                y_text,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    # Make room for annotations
    ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))

    fig.tight_layout()
    plt.show()


# %%
boxplot_per_celltype_sig(st_dict, "border_admixture_score")

# %% [markdown]
# ### Supervised module

# %% [markdown]
# The `sp` (supervised) module provides metrics to evaluate how well cell profiles
# in a spatial transcriptomics dataset agree with a reference single-cell RNA-seq
# (scRNA-seq) dataset with cell type annotations.
#
# Unlike scRNA-seq, contamination in spatial transcriptomics measurements mostly originates
# from the local tissue context.
#
# By comparing spatial expression profiles to a high-quality scRNA-seq reference,
# the supervised module aims to quantify this mismatch. Specifically, we compute metrics that measure:
# - how well each spatial cell matches its expected cell type,
# - how much its expression resembles other (neighboring) cell types, and
# - if it is possible to predict that a cell of one cell type is adjacent to a different cell type.
#
# To obtain cell-type specific marker genes, we define positive and negative markers in the
# annotated scRNA-seq via `markers_from_reference`.

# %% [markdown]
# *Computing cell-type specific markers*

# %%
markers = {}
for method, st in st_dict.items():
    markers[method] = st.markers_from_reference(
        adata_ref,
        ref_cell_type="celltype_major",
        ref_raw_counts_layer="raw",
        n_jobs=16,
    )

# %% [markdown]
# A list of positive and negative markers computed for each cell type is shown below.
# We will use xenium for demonstration.

# %%
for cell_type, sign in markers["xenium"].items():
    pos = sign["positive"]
    neg = sign["negative"]

    print(f"{cell_type}  |  positive: {', '.join(pos)}")
    print(f"{cell_type}  |  negative: {', '.join(neg)}\n")

# %% [markdown]
# Below, we show the number of negative markers that overlap with the positive markers of each cell type.
# To reliably estimate contamination, each cell type should share
# **at least ~5 negative markers** with the positive marker set of every other cell type.
# If these overlaps are too small, contamination estimates become unstable.
#
# In such cases, the marker definition can be relaxed by adjusting the thresholds used in
# `markers_from_reference` above.

# %%
ctypes = list(markers["xenium"].keys())
overlap_df = pd.DataFrame(0, index=ctypes, columns=ctypes, dtype=int)

for c in ctypes:
    neg_c = set(markers["xenium"][c].get("negative", []))
    for d in ctypes:
        pos_d = set(markers["xenium"][d].get("positive", []))
        overlap_df.loc[d, c] = len(neg_c & pos_d)

overlap_df

# %% [markdown]
# *Marker purity*
#
# To quantify how well each segmented cell in the spatial transcriptomics data
# matches its annotated cell type, we defined a marker-based purity (`marker_balanced_accuracy`)
# score that jointly evaluates the expression of positive (`positive_marker_recall`) and
# the absence of neighborhood-associated negative markers (`negative_marker_avoidance`).
# The method accounts for the spatial context of each cell and is motivated by the assumption
# that differences between scRNA-seq and spatial transcriptomics-derived cell type profiles
# arise mainly from local contamination by neighboring cells.

# %%
for method, st in st_dict.items():
    purity = st.sp.marker_purity(
        cell_type_key="transferred_cell_type",
        markers=markers[method],
        inplace=True,
    )

# %% [markdown]
# Proseg exhibits the highest `marker_balanced_accuracy` across most cell types, with the exception of DCIS1.
# To better interpret the overall purity scores, it is informative to examine the `positive_marker_recall`
# and `negative_marker_avoidance` scores separately, as they capture distinct aspects of marker specificity.

# %%
boxplot_per_celltype(st_dict, "marker_balanced_accuracy")

# %% [markdown]
# The `negative_marker_avoidance` score quantifies the extent to which a focal cell expresses positive
# markers of neighboring cell types, and thus reflects contamination. Overall, contamination levels are low;
# however, increased contamination is observed for DCIS1.


# %%
def box_strip_plot_per_celltype(
    st_dict,
    feature,
    celltype_col="transferred_cell_type",
    q=1.0,
    figsize=(15, 5),
    palette="Set2",
):
    dfs = []
    for method, st in st_dict.items():
        obs = st.sdata.tables["table"].obs
        tmp = obs[[celltype_col, feature]].copy()
        tmp = tmp[tmp[celltype_col].notna()]
        tmp["method"] = method
        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True).dropna(subset=[celltype_col, feature])

    if q < 1:
        df = df[df[feature] <= df[feature].quantile(q)]

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df,
        x=celltype_col,
        y=feature,
        hue="method",
        showcaps=True,
        showfliers=False,
        palette=palette,
        ax=ax,
    )

    sns.stripplot(
        data=df,
        x=celltype_col,
        y=feature,
        hue="method",
        dodge=True,
        jitter=0.2,
        alpha=0.6,
        linewidth=0,
        palette=palette,
        ax=ax,
    )

    # avoid duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="method")

    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Cell type")
    ax.set_ylabel(feature)
    ax.legend(title="method", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    plt.show()


# %%
box_strip_plot_per_celltype(st_dict, "negative_marker_avoidance")

# %% [markdown]
# In contrast, the `positive_marker_recall` score measures how consistently a cell
# expresses markers expected for its assigned cell type.

# %%
boxplot_per_celltype(st_dict, "positive_marker_recall")

# %% [markdown]
# These metrics are most informative when considered jointly. For example, we plot the mean
# `marker_balanced_accuracy` (a supervised measure of both how well positive markers are captured and negative
# markers are avoided) against the mean
# `border_admixture_score`, an unsupervised contamination metric. In this combined view,
# we can see that Proseg
# shows slighltly better performance than the other two methods.

# %%
rows = []
for _method, st in st_dict.items():
    corr_ncv_vs_center = st.sdata.tables["table"].obs["border_admixture_score"].median()
    negative_F1 = st.sdata.tables["table"].obs["marker_balanced_accuracy"].median()

    rows.append(
        {
            "Method": _method,
            "Border_admixture_score": corr_ncv_vs_center,
            "marker_balanced_accuracy": negative_F1,
        }
    )

results_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(4, 3))

sns.scatterplot(
    data=results_df,
    x="Border_admixture_score",
    y="marker_balanced_accuracy",
    hue="Method",
    style="Method",
    s=100,
    palette="Set2",
    ax=ax,
)

ax.set_xlabel("Border_admixture_score (↓)")
ax.set_ylabel("marker_balanced_accuracy (↑)")
ax.set_title("Border_admixture_score vs. marker_balanced_accuracy")

legend = ax.legend(title="Method", frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")

fig.tight_layout()
plt.show()

# %% [markdown]
# *Neighborhood contamination*
#
# Marker purity summarizes how well a cell matches its own markers and avoids neighborhood-relevant negatives.
# In many cases, we also want to quantify
# (i) **how many contaminating transcripts** are present per cell and
# (ii) **which neighboring cell types** contribute to this signal.
# We therefore compute neighborhood contamination.

# %%
for method, st in st_dict.items():
    per_cell_df, _, cont_bin, _ = st.sp.neighbor_contamination(
        cell_type_key="transferred_cell_type", markers=markers[method]
    )

# %% [markdown]
# Among the evaluated methods, Proseg yields the smallest number of cells showing detectable contamination.

# %%
methods_df = []

for method, st in st_dict.items():
    obs = st.sdata.tables["table"].obs
    counts = obs.loc[obs["contamination_counts"] > 0, "transferred_cell_type"].value_counts().rename(method)
    methods_df.append(counts)

merged = pd.concat(methods_df, axis=1)
merged

# %% [markdown]
# The spatial plot below shows the number of contaminating transcripts in each cell.


# %%
def plot_feature_labels(sdata, method, feature, axes, i):
    labels = sdata.tables["table"].obs["transferred_celltype_plot"].unique().astype(str).tolist()
    cols = [col_celltype[lab] for lab in labels]

    sdata.tables["table"].obs["region"] = "cell_boundaries"
    sdata.set_table_annotates_spatialelement("table", region="cell_boundaries")

    sdata.pl.render_shapes(
        "cell_boundaries",
        color="transferred_celltype_plot",
        palette=cols,
        groups=labels,
        outline_color="white",
        outline_width=0.5,
    ).pl.show(ax=axes[0, i], title=f"{method}:Cell boundaries colored by cell type", coordinate_systems="global")

    sdata.pl.render_shapes(
        "cell_boundaries",
        color=feature,
        outline_color="white",
        outline_width=0.5,
    ).pl.show(ax=axes[1, i], title=f"{method}: {feature}", coordinate_systems="global")


# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
axes = np.atleast_2d(axes)
for i, (method, st) in enumerate(st_dict.items()):
    plot_feature_labels(st.sdata, method, "contamination_counts", axes, i)

# %% [markdown]
# The heatmap below summarizes contamination strength for each source–target cell-type pair.
# Each entry represents the mean contamination strength across all evaluable target cells
# of the given target cell type, where the source-specific contamination strength is computed
# as the fraction of transcripts in the target cell that correspond to contamination-relevant
# markers of the source cell type.
#
# The bubble plot illustrates the contamination strength (bubble color) and the
# number of evaluable target cells (bubble size).

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize

vmax = max(st.sdata.tables["table"].uns["contamination_strength_matrix"].max().max() for st in st_dict.values())
norm = Normalize(vmin=0, vmax=vmax)

cell_type_order = list(next(iter(st_dict.values())).sdata.tables["table"].uns["contamination_strength_matrix"].columns)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

for i, (method, st) in enumerate(st_dict.items()):
    cont_strength = st.sdata.tables["table"].uns["contamination_strength_matrix"]
    cont_n = st.sdata.tables["table"].uns["contamination_evaluable_cells_matrix"]

    # Force same source/target order and include missing rows/columns.
    cont_strength = cont_strength.reindex(
        index=cell_type_order,
        columns=cell_type_order,
    )
    cont_n = cont_n.reindex(
        index=cell_type_order,
        columns=cell_type_order,
    )

    plot_df = (
        cont_strength.stack(dropna=False)
        .rename("contamination_strength")
        .reset_index()
        .rename(columns={"level_0": "source", "level_1": "target"})
    )

    plot_df["n_evaluable"] = cont_n.stack(dropna=False).values
    plot_df = plot_df.dropna(subset=["n_evaluable"])

    plot_df["target"] = pd.Categorical(
        plot_df["target"],
        categories=cell_type_order,
        ordered=True,
    )
    plot_df["source"] = pd.Categorical(
        plot_df["source"],
        categories=cell_type_order,
        ordered=True,
    )

    sns.scatterplot(
        data=plot_df,
        x="target",
        y="source",
        size="n_evaluable",
        hue="contamination_strength",
        sizes=(20, 600),
        palette="Reds",
        hue_norm=norm,
        edgecolor="black",
        legend=False,
        ax=axes[i],
    )

    axes[i].set_title(method, fontsize=12)
    axes[i].set_xlabel("Target Cell Type")
    axes[i].set_xlim(-0.5, len(cell_type_order) - 0.5)
    axes[i].set_ylim(len(cell_type_order) - 0.5, -0.5)

    axes[i].set_xticks(range(len(cell_type_order)))
    axes[i].set_xticklabels(cell_type_order, rotation=45, ha="right")

    axes[i].set_yticks(range(len(cell_type_order)))
    if i == 0:
        axes[i].set_ylabel("Source Cell Type")
        axes[i].set_yticklabels(cell_type_order)
    else:
        axes[i].set_ylabel("")
        axes[i].set_yticklabels([])

plt.show()

# %% [markdown]
# *Mutually exclusive co-expression rate (MECR)*

# %% [markdown]
# The mutually exclusive co-expression rate (MECR) is a measure for whether
# combinations of positive and negative markers (computed with a more stringent setting
# to increase mutual exclusivity, `vote_frac_pos=0.3`) co-occur less often than
# expected under independence (using Fisher's exact test). By conditioning on the
# marginal detection frequencies of each gene, Fisher’s exact test does not favor
# methods with low overall transcript counts.

# %%
markers = {}

for method, st in st_dict.items():
    markers[method] = st.markers_from_reference(
        adata_ref,
        ref_cell_type="celltype_major",
        min_pos_frac=0.3,
        ref_raw_counts_layer="raw",
        n_jobs=16,
    )

    mecr = st.sp.mutually_exclusive_coexpression_rate(
        markers=markers[method],
    )

# %% [markdown]
# Across all methods, marker pairs with significant mutual exclusivity show substantially
# lower co-expression than expected under independence.
# Although the differences are minor, ProSeg shows the strongest depletion of co-expression
# among these marker pairs.

# %%
rows = []

for method, st in st_dict.items():
    tbl = st.sdata.tables["table"]
    mecr_df = tbl.uns["mutually_exclusive_coexpression_rate"]

    # Keep marker pairs with significant mutual exclusivity
    df_sig = mecr_df.loc[
        mecr_df["odds_ratio"].notna()
        & np.isfinite(mecr_df["odds_ratio"])
        & mecr_df["pvalue"].notna()
        & np.isfinite(mecr_df["pvalue"])
        & (mecr_df["odds_ratio"] < 1)
        & (mecr_df["pvalue"] < 0.05)
    ].copy()

    rows.extend(
        {
            "method": str(method),
            "Fisher_OR": row["odds_ratio"],
        }
        for _, row in df_sig.iterrows()
    )

df = pd.DataFrame(rows)

# Order methods by mean OR
mean_order = (
    df.groupby("method")["Fisher_OR"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

means = (
    df.groupby("method")["Fisher_OR"]
    .mean()
    .reindex(mean_order)
)

xtick_labels = [
    f"{m}\nmean: {means[m]:.2f}"
    for m in mean_order
]

plt.figure(figsize=(6, 4))

ax = sns.violinplot(
    data=df,
    x="method",
    y="Fisher_OR",
    order=mean_order,
    palette="Set2",
    linewidth=2,
)

sns.stripplot(
    data=df,
    x="method",
    y="Fisher_OR",
    order=mean_order,
    color="black",
    size=2.5,
    alpha=0.35,
    jitter=0.25,
    ax=ax,
)

# OR = 1 corresponds to independence
ax.axhline(
    1,
    ls="--",
    lw=1,
    color="gray",
    alpha=0.6,
)

ax.set_xticklabels(xtick_labels)

ax.set_ylabel("Mutually exclusive marker co-expression (odds ratio)")
ax.set_xlabel("")
ax.set_title(
    "Significantly mutually exclusive marker pairs"
)

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3D Volume Module
#
# The volume (`vl`) accessor provides metrics to assess how well a segmentation method
# resolves cell overlaps in 3D. Spatial transcriptomics tissue sections have a finite
# thickness (~4–10 µm), so cells can overlap along the z-dimension and 2D segmentation
# methods may introduce mixing by assigning transcripts from overlapping cells to the same mask.
# In this module, we introduce metrics to quantify sensitivity to 3D overlap and evaluate how
# well quasi-3D methods (e.g. Proseg) disentangle transcripts from overlapping cells.
#
# For a detailed description of this module, please refer to this [tutorial](volume.ipynb).

# %% [markdown]
# *Top-bottom z consistency*
#
# To detect potential z-overlap mixing within segmented cells, we compare gene expression profiles between the
# bottom and top z-regions of each cell. `similarity_top_bottom` reports the observed cosine similarity relative
# to its permutation-based null expectation, with negative residuals indicating lower consistency across z than
# expected. In the histogram below, we show these residuals only for cells with unexpectedly low top–bottom
# similarity (`similarity_top_bottom_p_value` < 0.05).

# %%
for _method, st in st_dict.items():
    _cos_sim = st.vl.similarity_top_bottom()

# %% [markdown]
# Proseg (a quasi-3D method) shows the fewest cells with unexpectedly different expression profiles between the
# top and bottom of the cell (`similarity_top_bottom_p_value` < 0.05). Moreover, even among these cells, the
# differences between top and bottom are generally smaller than for Xenium and BIDCell, suggesting more
# consistent transcript composition across z.

# %%
histogram_feature(st_dict, "similarity_top_bottom")

# %% [markdown]
# In [volume.ipynb](./volume.ipynb), we explore the distribution of transcripts along the z-dimension
# in more depth and put it into context with
# [ovrlpy](https://www.biorxiv.org/content/10.1101/2025.01.13.632601v2.full),
# a package for detecting 3D overlap in spatial transcriptomics.

# %% [markdown]
# ### Point statistic metrics
#
# The point statistics (`ps`) module is designed to compare the
# distribution of a set of transcripts in the cell relative to its cell centroid or cell border.
# The idea is to compute the distances of the transcripts to a reference point in the cell,
# either the cell centroid or the cell boundaries and aggregate this measure per transcript id.
#
# _distance to membrane metric_
#
# In this metric we compute the distance to the segmented cell membrane of each transcript coordinate
# and aggregate this metric per transcript id as mean.

# %%
plt.style.use("dark_background")


def plot_metric(st, method, metric, genes, ax):
    sdata_plot = sd.deepcopy(st.sdata)
    sdata_plot[st.tables_key] = sdata_plot[st.tables_key][(sdata_plot[st.tables_key].obs[metric].notna())]

    sdata_plot.tables[st.tables_key].obs["region"] = st.shapes_key
    sdata_plot.set_table_annotates_spatialelement(st.tables_key, region=st.shapes_key)

    sdata_plot.pl.render_shapes(
        element=st.nucleus_shapes_key,
        fill_alpha=0.2,
        outline_alpha=1.0,
        outline_width=0.5,
        outline_color="black",
    ).pl.render_shapes(
        element=st.shapes_key,
        color=metric,
        cmap="viridis",
        fill_alpha=0.5,
        outline_alpha=1.0,
        outline_width=0.5,
        outline_color="black",
    ).pl.render_points(
        st.points_key,
        color=st.points_gene_key,
        groups=genes,
        palette=["red"] * len(genes),
    ).pl.show(ax=ax, title="Cell/nuclei seg. for " + method + " " + metric, colorbar=True)

    return fig, axes


# %% [markdown]
# First, we compute both the average distance to the cell membrane across all previously
# defined negative and positive markers for the cell type "DCIS1".

# %%
border_distance_negative = {}
border_distance_positive = {}

for method, st in st_dict.items():
    border_distance_negative[method] = st.ps.distance_to_membrane(
        markers[method]["DCIS1"]["negative"],
        cell_type_key="transferred_celltype_plot",
        cell_type_query=["DCIS1"],
        restrict_to_within_boundary=True,
        inplace=False,
    )
    border_distance_positive[method] = st.ps.distance_to_membrane(
        markers[method]["DCIS1"]["positive"],
        cell_type_key="transferred_celltype_plot",
        restrict_to_within_boundary=True,
        cell_type_query=["DCIS1"],
        inplace=False,
    )

# %%
keys = []
values = []

for key, df in border_distance_negative.items():
    for val in df["distance_to_cell_membrane_norm_166_genes"]:
        keys.append(key)
        values.append(val)

result_negative = pd.DataFrame({"method": keys, "distance": values, "marker": "negative"})

keys = []
values = []

for key, df in border_distance_positive.items():
    for val in df["distance_to_cell_membrane_norm_78_genes"]:
        keys.append(key)
        values.append(val)

result_positive = pd.DataFrame({"method": keys, "distance": values, "marker": "positive"})

# %%
result = pd.concat([result_negative, result_positive])
result = result.dropna()

# %%
plt.figure(figsize=(6, 4))

ax = sns.violinplot(
    data=result,
    x="method",
    y="distance",
    hue="marker",
    palette="Set2",
    linewidth=2,
)

ax.set_ylabel("distance to membrane")
ax.set_xlabel("")
ax.set_title("Distance to membrane across cells for DCIS1")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# We can plot now the distance to membrane for the positive and negative markers
# across the different segmentation algorithms and for the cell type "DCIS1".

# %% [markdown]
# We can also run the method inplace and plot the results in space

# %%
for _method, st in st_dict.items():
    st.ps.distance_to_membrane(
        markers[method]["DCIS1"]["negative"],
        cell_type_key="transferred_celltype_plot",
        cell_type_query=["DCIS1"],
        restrict_to_within_boundary=True,
        inplace=True,
    )

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 10), constrained_layout=True)

for i, (method, st) in enumerate(st_dict.items()):
    plot_metric(
        st,
        method,
        "distance_to_cell_membrane_norm_166_genes",
        markers[method]["DCIS1"]["negative"][:20],
        axes[i],
    )

# %% [markdown]
# We can identify cells in which the distribution of negative markers is clearly
# more skewed than for thers and that these values differ again greatly between the segmentation algorithms.

# %% [markdown]
# ### Metrics orthogonolisation

# %% [markdown]
# In order to look at the co-linearity of the computed metrics,
# we will compute the correlation matrix (scaled covariance matrix) of all the metrics.
# Since the directionality (positively or negatively correlated) does not matter,
# we take the absolute value of the correlation coefficient as our orthogonality metric.
# Like this, we can visualise which metrics correlate

# %%
# code adapted from claude.ai
for method, st in st_dict.items():
    df = st.sdata.tables["table"].obs
    df_numeric = df.select_dtypes(include="number")
    df = pd.DataFrame(df_numeric, columns=df_numeric.columns, index=df_numeric.index)
    # correlation does not work with NA values. Fill NA with columnwise mean
    df = df.apply(lambda x: x.fillna(x.mean()), axis=0)
    correlation_matrix = df.corr()

    # take absolute value instead - sign is not important, only co-linearity
    correlation_matrix = correlation_matrix.abs()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=0,
        vmax=1,
        xticklabels=correlation_matrix.columns,
        yticklabels=correlation_matrix.columns,
    )
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title(f"Absolute Correlation Matrix Heatmap {method}")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# There are quite some metrics which correlate with the segmented cell area,
# such as total counts of all transcripts and metrics related to this.
# We note as well that the correlation of metrics differs between algorithms.

# %% [markdown]
# ## Session Info

# %%
print(sd.__version__)  # spatialdata
print(spatialdata_plot.__version__)
