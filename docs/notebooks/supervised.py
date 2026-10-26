# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: allen (3.13.5)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Module: Supervised Metrics
# 
# Unlike scRNA-seq, spatial transcriptomics measurements are affected by the local tissue context.
# Transcripts can be captured outside their cell of origin due to transcript diffusion, 3D overlap,
# or segmentation inaccuracies, leading to mixed or contaminated expression profiles.
# As a result, spatial cells may partially resemble their neighbors.
# 
# <center>
#  <img src='../_static/img/docs/supervised.png' width='90%' />
# </center>
# 
# The `supervised` (`sp`) module provides metrics to evaluate how well cell profiles in a spatial
# transcriptomics dataset agree with a reference single-cell RNA-seq (scRNA-seq) dataset
# with cell type annotations.
# 
# By comparing spatial expression profiles to a high-quality scRNA-seq reference,
# the supervised module aims to quantify this mismatch. Specifically,
# we use the scRNA-seq dataset as a clean reference, transfer cell type labels to the spatial data,
# and then compute metrics that measure:
# - how well each spatial cell matches its expected cell type,
# - how much its expression resembles other (neighboring) cell types, and
# - if it is possible to predict that a cell of one cell type is adjacent to a different cell type.
# 
# To follow along with this tutorial, you can download the data from [here](https://oc.embl.de/index.php/s/YSvZTt8AArh4c5a).

# %%
# %load_ext autoreload
# %autoreload 2

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# %%
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spatialdata as sd
import spatialdata_plot  # noqa

import segtraq

segtraq.settings.n_jobs = 8  # Use 8 CPU cores (-1 for all available cores)

# %% [markdown]
# #### Transfer labels from scRNA-seq to spatial transcriptomics
# 
# We first load a spatial transcriptomics dataset (stored as a `SpatialData` object)
# and an annotated scRNA-seq reference dataset (stored as an `AnnData` object).
# For demonstration purposes, we subset the spatial dataset to a smaller region to reduce runtime.
# 
# Next, we initialize a `SegTraQ` object from the SpatialData. During initialization,
# the `validate_spatialdata()` method is automatically run to ensure that all required attributes
# (e.g. `images_key`, `tables_key`) are correctly specified.
# 
# We then transfer cell type labels from the scRNA-seq reference to the spatial dataset
# using `run_label_transfer()`. This method computes cell type–specific mean expression
# profiles in the reference and assigns each spatial cell to the cell type with the highest
# Pearson correlation to its expression profile.
# 
# Finally, we define a color scheme for visualizing the transferred cell type labels.

# %%
# Load spatial transcriptomics dataset
sdata_ws = sd.read_zarr("../../data/xenium_5K_data/xenium.zarr")

# Subset the dataset to a specific bounding box
bb_xmin = 800
bb_ymin = 1150
bb_w = 200
bb_h = 300
bb_xmax = bb_xmin + bb_w
bb_ymax = bb_ymin + bb_h

sdata = sdata_ws.query.bounding_box(
    axes=["x", "y"],
    min_coordinate=[bb_xmin, bb_ymin],
    max_coordinate=[bb_xmax, bb_ymax],
    target_coordinate_system="global",
)

# initialize SegTraQ object
st = segtraq.SegTraQ(sdata, images_key="image", tables_centroid_x_key="x_centroid", tables_centroid_y_key="y_centroid")

# Load scRNA-seq dataset
scRNAseq_data_path = Path("../../data/xenium_5K_data/BC_scRNAseq_Janesick.h5ad")
adata_ref = ad.read_h5ad(scRNAseq_data_path)

# Define color palette for cell types
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
# The transferred labels are stored in `sdata.tables[tables_key].obs["transferred_cell_type"]`.
# The column name can be changed via the `cell_type_key` argument.
# 
# By default, cells with **≤ 10 or ≥ 2000 transcripts** and **≤ 5 detected genes** are filtered out.
# These thresholds can be adjusted using `tx_min`, `tx_max`, `gn_min`, and `gn_max`.
# 
# Ensure that gene identifiers match between `adata_ref.var_names` and
# `sdata.tables[tables_key].var_names`. If they differ, specify the corresponding
# identifier columns using `ref_ensemble_key` and `query_ensemble_key`.
# If these are left as `None` (default), `var_names` are used.

# %%
st.run_label_transfer(
    adata_ref,
    ref_cell_type="celltype_major",  # Reference cell type column
    ref_raw_counts_layer="raw",  # Reference raw counts layer
)

# %% [markdown]
# Next, cell shapes can be plotted and colored by cell type based on a defined color palette.
# Some cells lack a transferred cell-type label (due to filtering low-count cells before label transfer);
# for compatibility with `spatialdata-plot`, we replace missing values (`NaN`) with "Unknown" before plotting.

# %%
# Replace NaN with Unknown
s = st.sdata.tables["table"].obs["transferred_cell_type"]
if pd.api.types.is_categorical_dtype(s):
    s = s.cat.add_categories(["Unknown"])

st.sdata.tables["table"].obs["transferred_cell_type_plot"] = s.fillna("Unknown")

labels = st.sdata.tables["table"].obs["transferred_cell_type_plot"].unique().astype(str).tolist()
cols = [col_celltype[lab] for lab in labels]

# %%
axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)[1].flatten()

# Dapi image
st.sdata.pl.render_images("image").pl.show(ax=axes[0], title="DAPI image", coordinate_systems="global")

# Add link between table and spatial element
st.sdata.tables["table"].obs["region"] = "cell_boundaries"
st.sdata.set_table_annotates_spatialelement("table", region="cell_boundaries")

st.sdata.pl.render_shapes(
    "cell_boundaries",
    color="transferred_cell_type_plot",
    palette=cols,
    groups=labels,
    outline_color="white",  # outlines visible on black
    outline_width=0.5,
).pl.show(ax=axes[1], title="Cell boundaries", coordinate_systems="global")

# %% [markdown]
# #### Computing cell type-specific markers

# %% [markdown]
# **Positive markers** are genes that are consistently upregulated in one cell type relative to other cell types.
# To avoid composition bias arising from unequal cell type abundances, we perform
# **pairwise comparisons between all cell types** in the reference dataset.
# Marker detection can be performed using either differential expression (`mode="de"`) or
# AUC-based scoring (`mode="auc"`).
# 
# For each cell type, genes are selected by a voting scheme: a gene must be identified as
# upregulated in at least a fraction of pairwise contrasts (`vote_fraction_pos`, default = 0.5).
# To ensure that positive markers are representative of the cell type rather than driven
# by a small subset of cells, we additionally require that a gene is expressed (> 0) in at least
# `min_pos_frac` (default = 0.1) of cells of that type. To further increase specificity,
# we require that a positive marker is not shared by more than `t_pos × n_celltypes` cell types.
# This yields markers that are both upregulated and cell-type–specific,
# without bias from cell type composition in the reference data.
# 
# **Negative markers** are derived from the same pairwise contrasts but are defined as genes that
# are effectively off in a focal cell type while being upregulated elsewhere. For each ordered pair of
# cell types `(a, b)`, genes that are upregulated in `a` relative to `b` are considered negative-marker
# candidates for `b` if they are expressed (`> 0`) in at most `max_neg_frac` (default = 0.05)
# of cells of type b. To avoid ambiguous genes with inconsistent directionality,
# we further require that a negative marker of `b` is not identified as upregulated in `b` relative
# to any other cell type across all ordered contrasts.
# 
# When using `mode="auc"`, the minimum required AUC can be set via `auc_pos_thresh` (default = 0.9).
# When using `mode="de"`, the differential expression method can be selected via `method`
# (`"wilcoxon"`, `"t-test"`, or `"logreg"`), with thresholds defined by `logFC_pos_thresh`
# (default = 1.0) and `pval_adj_thresh` (default = 0.05).
# 
# All computations can be parallelized by specifying the number of jobs via `n_jobs`.
# By default, `markers_from_reference()` stores the identified positive and negative markers in
# `sdata.tables["table"].uns["segtraq_markers"]`. Markers are stored as gene indices to minimize
# memory usage and are automatically used by the supervised metrics when `markers=None`.
# Alternatively, markers can be passed explicitly to the individual metrics.

# %%
# get cell type-specific markers from the reference dataset
markers = st.markers_from_reference(
    adata_ref,
    ref_cell_type="celltype_major",
    ref_raw_counts_layer="raw",
    mode="de",
    vote_fraction_pos=0.5,
    min_pos_frac=0.1,
    max_neg_frac=0.05,
    n_jobs=16,
)

# %% [markdown]
# A list of positive and negative markers computed for each cell type is shown below.

# %%
for cell_type, sign in markers.items():
    pos = sign["positive"]
    neg = sign["negative"]

    print(f"{cell_type}  |  positive: {', '.join(pos)}")
    print(f"{cell_type}  |  negative: {', '.join(neg)}\n")

# %% [markdown]
# Below, we show the number of negative markers that overlap with the positive markers of each cell type.
# To reliably estimate contamination, each cell type should share **at least ~5 negative markers**
# with the positive marker set of every other cell type. If these overlaps are too small,
# contamination estimates become unstable.
# 
# In such cases, the marker definition can be relaxed by adjusting the thresholds used in
# `markers_from_reference` above.

# %%
ctypes = list(markers.keys())
overlap_df = pd.DataFrame(0, index=ctypes, columns=ctypes, dtype=int)

for c in ctypes:
    neg_c = set(markers[c].get("negative", []))
    for d in ctypes:
        pos_d = set(markers[d].get("positive", []))
        overlap_df.loc[d, c] = len(pos_d & neg_c)

overlap_df

# %% [markdown]
# #### Compute marker balanced accuracy

# %% [markdown]
# We quantify how well each spatial cell’s expression profile matches its assigned
# cell type using `marker_balanced_accuracy()`, which combines positive-marker
# detection with the absence of neighborhood-relevant negative markers.
# 
# Local contamination by neighboring cells (e.g. diffusion or segmentation spillover)
# can contribute to differences between reference and spatial expression profiles.
# This motivates evaluating negative markers in a neighborhood-aware manner.
# 
# **Positive-marker detection**
# 
# For each cell of type `c`, we compute the fraction of positive marker genes
# (`markers[c]["positive"]`) that are expressed in the cell (`> 0`).
# This measures how consistently the cell expresses markers expected for its
# assigned type and is computed independently of the neighborhood.
# 
# **Negative-marker avoidance (neighborhood-aware)**
# 
# For a focal cell, we identify the cell types present in its spatial neighborhood and
# intersect the focal cell type’s negative markers with the positive markers of neighboring types.
# Optionally, these markers can be further restricted to genes expressed in at least one
# neighboring cell of the corresponding source type (`require_neighbor_expression=True`).
# We then compute the fraction of these relevant negative markers that are not
# expressed in the focal cell.
# 
# If no neighborhood graph is present (`neighbors_key not in sdata.tables[tables_key].obsp`),
# it is automatically computed from cell centroids using a Delaunay graph.
# 
# **Marker balanced accuracy**
# 
# The two fractions are averaged:
# 
# `marker_balanced_accuracy = 0.5 * (positive-marker detection + negative-marker avoidance)`
# 
# This corresponds to the mean of sensitivity and specificity. High values indicate
# consistent detection of expected markers together with little expression of
# neighborhood-relevant negative markers.
# 
# The two components are calculated internally but are not returned separately.
# The output contains `marker_balanced_accuracy`, `n_evaluated_positive_markers`,
# and `n_evaluated_negative_markers`, together with the cell identifiers.

# %%
purity = st.sp.marker_balanced_accuracy(
    cell_type_key="transferred_cell_type",
    require_neighbor_expression=True,
    neighbors_key="spatial_connectivities",
)

# %% [markdown]
# ##### `negative_marker_avoidance`

# %%
def boxplot_per_celltype(
    sdata,
    feature,
    celltype_col="transferred_cell_type",
    q=1.0,
    figsize=(10, 5),
    ascending=True,
    palette=None,
):
    obs = sdata.tables["table"].obs
    df = obs[[celltype_col, feature]].dropna().copy()

    if q < 1:
        df = df[df[feature] <= df[feature].quantile(q)]

    # order by median (low -> high). Use .sort_values(ascending=False) for high -> low
    order = df.groupby(celltype_col)[feature].median().sort_values(ascending=ascending).index.tolist()

    # ensure palette covers all categories (optional safety)
    if palette is not None:
        palette = {ct: palette.get(ct, "#808080") for ct in order}

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df,
        x=celltype_col,
        y=feature,
        order=order,
        hue=celltype_col,
        palette=palette,
        showcaps=True,
        showfliers=False,
        legend=False,
        ax=ax,
    )

    sns.stripplot(
        data=df,
        x=celltype_col,
        y=feature,
        order=order,
        hue=celltype_col,
        palette=palette,
        jitter=0.25,
        dodge=False,
        alpha=0.6,
        edgecolor="black",
        linewidth=0.4,
        legend=False,
        ax=ax,
    )

    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Cell type")
    ax.set_ylabel(feature)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ##### `marker_balanced_accuracy`

# %% [markdown]
# **DCIS1** ranks among the cell types with the lowest marker balanced accuracy,
# while **DCIS2** and **mast** cells show the highest marker balanced accuracy.

# %%
boxplot_per_celltype(st.sdata, feature="marker_balanced_accuracy", palette=col_celltype)

# %% [markdown]
# #### Contamination strength

# %% [markdown]
# Marker balanced accuracy summarizes how well a cell matches its own markers and avoids
# neighborhood-relevant negatives. 
# In many cases, we also want to quantify
# (i) **what fraction of a cell's transcripts can be attributed to contamination** and
# (ii) **which neighboring cell types** contribute to this signal.
# We use `contamination_strength()` to quantify potential local contamination.
# 
# A gene is contamination-relevant for a target cell of type `c_tgt` if it is:
# (i) a **negative marker of `c_tgt`**,
# (ii) a **positive marker of a neighboring source type `c_src`**, and
# (iii) **expressed (>0) in at least one neighboring cell of that source type**
# when `require_neighbor_expression=True` (the default).
# 
# The function returns three outputs:
# 
# **Per-cell contamination strength**
# `contamination_strength` is the fraction of transcripts assigned to a cell
# that correspond to contamination-relevant genes. 
# With `inplace=True`, the score is stored in `.obs`.
# 
# **Directed cell-type contamination matrices**
# Rows represent source cell types and columns represent target cell types:
# - `contamination_strength_matrix`: the mean source-specific contamination
#   strength across evaluable target cells with nonzero total counts,
#   including cells with zero contamination-relevant transcripts.
# - `contamination_evaluable_cells_matrix`: the number of target cells for
#   which contamination from each source type could be evaluated.
# 
# With `inplace=True`, both matrices are stored in `.uns`.
# 
# If no neighborhood graph is present (`neighbors_key not in sdata.tables[tables_key].obsp`),
# it is automatically computed from cell centroids using a Delaunay graph.

# %%
_, _, _ = st.sp.contamination_strength(
    cell_type_key="transferred_cell_type",
    require_neighbor_expression=True,
    neighbors_key="spatial_connectivities",
    inplace=True,
)

# %% [markdown]
# ##### `contamination_strength`

# %% [markdown]
# We can visualize `contamination_strength`, the fraction of transcripts
# assigned to a cell that correspond to locally relevant contamination marker genes.
# A value of contamination_strength = 0 indicates that no contamination-relevant
# transcripts were detected, whereas larger values indicate that a greater proportion of
# the cell's assigned transcripts are consistent with potential contamination from neighboring cells.

# %%
axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)[1].flatten()

st.sdata.pl.render_shapes(
    "cell_boundaries",
    color="transferred_cell_type_plot",
    palette=cols,
    groups=labels,
    outline_color="white",
    outline_width=0.5,
).pl.show(ax=axes[0], title="Cell boundaries colored by cell type", coordinate_systems="global")

st.sdata.pl.render_shapes(
    "cell_boundaries",
    color="contamination_strength",
    outline_color="white",
    outline_width=0.5,
).pl.show(ax=axes[1], title="Cell boundaries colored by contamination strength", coordinate_systems="global")

# %% [markdown]
# ##### `.uns["contamination_strength_matrix"]`

# %% [markdown]
# The heatmap below summarizes contamination strength for each source–target cell-type pair.
# Each entry represents the mean contamination strength across all evaluable target cells of the
# given target cell type, where the source-specific contamination strength is computed as the
# fraction of transcripts in the target cell that correspond to contamination-relevant markers of the source cell type.
# 
# The bubble plot of the `contamination_strength_matrix` provides complementary information by quantifying
# how large the contamination signal is when it occurs. For example, although myoepithelial and DCIS2 cells
# are frequently neighbors, contamination from DCIS2 into myoepithelial cells is negligible.
# Moreover, relatively few myoepithelial cells are evaluable for DCIS2 despite their spatial
# proximity because the two cell types share only a small number of mutually exclusive markers
# (four in this example; see `overlap_df`), and these markers may not be expressed in the
# neighboring DCIS2 cells in every local neighborhood.
# 
# This can have both technical and biological causes. For example, when two cell types have similar
# expression profiles, contamination between them is less consequential because the
# contaminating transcripts largely match the expected expression of the target cell.
# 

# %%
cont_strength_mat = st.sdata.tables[st.tables_key].uns["contamination_strength_matrix"]
cont_n = st.sdata.tables[st.tables_key].uns["contamination_evaluable_cells_matrix"]

plot_df = (
    cont_strength_mat.stack(dropna=False)
    .rename("contamination_strength")
    .reset_index()
    .rename(columns={"level_0": "source", "level_1": "target"})
)

plot_df["n_evaluable"] = cont_n.stack(dropna=False).values

plt.figure(figsize=(9, 7))

ax = sns.scatterplot(
    data=plot_df,
    x="target",
    y="source",
    size="n_evaluable",
    hue="contamination_strength",
    sizes=(20, 600),
    palette="Reds",
    edgecolor="black",
)

ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0,
)

plt.title("Directed Cell-Type Contamination Strength")
plt.xlabel("Target Cell Type")
plt.ylabel("Source Cell Type")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# #### **! Important**
# The observed contamination between two cell types is influenced by three main factors:
# 
# 1) **True contamination**, i.e. transcript spillover or transcript misassignment between neighboring cells.
# 2) **Spatial proximity**, i.e. how frequently the two cell types occur as neighbors.
# The more often they are adjacent, the greater the opportunity for contamination.
# 3) **Expression similarity**, .ie. contamination between transcriptionally similar
# cell types is both less consequential and more difficult to detect.
# Because the two cell types share fewer mutually exclusive marker genes,
# transferred transcripts have less impact on the measured expression profile,
# and fewer informative genes are available for quantifying contamination.

# %% [markdown]
# #### Compute RCTD contamination

# %% [markdown]
# We estimate expression mixture within each spatial cell using
# `rctd_contamination()` and an annotated scRNA-seq reference.
# Unlike the marker-based metrics, this approach does not require assigned
# spatial cell-type labels, predefined marker sets, or neighborhood information.
# 
# **Reference-based expression decomposition**
# 
# RCTD is run in full mode to estimate contributions from all retained reference
# cell types to each spatial cell's expression profile. Query and reference
# expression matrices must contain raw counts.
# 
# **RCTD contamination**
# 
# For each spatial cell, the fitted cell-type weights are normalized to sum to one.
# The score is then calculated as:
# 
# `rctd_contamination = 1 - max(normalized_weights)`
# 
# This represents the estimated expression contribution outside the dominant
# reference cell type. For example, normalized weights of 0.8 and 0.2 yield
# a score of 0.2, regardless of the spatial cell's assigned label.
# 
# Lower values indicate that a single reference type dominates the fitted
# expression profile; higher values indicate a greater mixture.
# With K retained reference types, the maximum possible score is `1 - 1 / K`.
# 
# The score measures reference-dependent expression mixture rather than directly
# identifying incorrectly assigned transcripts. 
# 
# The output contains cell identifiers and `rctd_contamination`.
# Cells excluded by RCTD's filters, non-converged fits, and invalid fits receive NaN.
# With `inplace=True` (the default), the score is also stored in `.obs`.
# 
# RCTD was developed by [Cable et al. (2022)](https://doi.org/10.1038/s41587-021-00830-w).
# SegTraQ uses [rctd-py](https://github.com/p-gueguen/rctd-py), the Python
# implementation.

# %%
rctd_df = st.sp.rctd_contamination(
    adata_ref=adata_ref,
    ref_cell_type="celltype_major",
    ref_raw_counts_layer="raw",
    inplace=True,
)

# %% [markdown]
# ##### `rctd_contamination`
# 
# Comparing score distributions across cell types highlights differences in
# estimated expression mixture. Interpret these alongside marker balanced
# accuracy and contamination strength to assess whether mixture is accompanied
# by marker mismatch and evidence of local contamination.

# %%
boxplot_per_celltype(st.sdata, feature="rctd_contamination", palette=col_celltype)

# %% [markdown]
# #### Compute mutually exclusive co-expression rate (MECR)
# 
# The `mutually_exclusive_coexpression_rate` function identifies marker-gene
# pairs that are expected to be mutually exclusive based on a reference dataset.
# Candidate pairs are derived from reciprocal positive/negative marker
# relationships and are retained only if their observed co-expression in the
# reference is sufficiently below that expected under independence.
# 
# For these reference-defined mutually exclusive pairs, co-expression in the
# spatial data is evaluated using Fisher's exact test on binary gene detection
# (expression > 0). A one-sided test with `alternative="greater"` asks whether
# the two genes are positively associated in the spatial data.
# 
# Thus, an odds ratio > 1 indicates increased co-expression of a marker pair
# that is expected to be mutually exclusive based on the reference.

# %%
mecr = st.sp.mutually_exclusive_coexpression_rate(
    adata_ref=adata_ref,
    ref_cell_type="celltype_major",
    ref_raw_counts_layer="raw",
    inplace=True,
)

# %% [markdown]
# This tutorial subset contains only 751 cells, while 2,248 marker pairs are
# tested. Consequently, many pairs are very sparse and individual Fisher tests
# have limited power after multiple-testing correction. After correction for
# testing many gene pairs, no pair remains significant at the selected FDR threshold.

# %%
mecr_plot = mecr.copy()

mecr_plot = mecr_plot.loc[(mecr_plot["odds_ratio"] > 1) & (mecr_plot["pvalue_adj"] < 0.05)]

mecr_plot

# %% [markdown]
# ## Session Info

# %%
print(sd.__version__)  # spatialdata
print(spatialdata_plot.__version__)


