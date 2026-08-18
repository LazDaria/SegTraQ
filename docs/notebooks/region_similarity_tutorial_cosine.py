# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: segtraq_26 (3.11.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Module: Region Similarity
#
# Assuming that transcripts are homogeneously distributed throughout the cell,
# we expect there to be similar expression of genes in the nucleus and in the rest of the cell.
# If this is not the case, it can be indicative of transcript spillover from adjacent cells.
#
# <center>
#  <img src='../_static/img/docs/region_similarity.png' width='90%' />
# </center>
#
# The `region similarity` (`rs`) module compares transcript-composition profiles between
# subcellular regions (e.g., cell and nucleus). Count profiles are transformed with PFlog1pPF
# (shifted CLR) and compared using cosine similarity. To account for finite-count effects, the
# reported score is a permutation-corrected residual: observed cosine similarity minus the mean
# similarity expected under a conditional random-splitting null.
#
# Residuals around zero are close to the null expectation, negative values indicate lower similarity
# than expected, and positive values indicate higher similarity than expected. The accompanying
# lower-tail permutation p-value quantifies evidence for unusually low similarity.
#
# To follow along with this tutorial, you can download the data from [here](https://oc.embl.de/index.php/s/iGxVy8qtZnwHOju).

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spatialdata as sd
import spatialdata_plot  # noqa
from scipy.stats import false_discovery_control, linregress

import segtraq

# %% [markdown]
# We start out by loading the data into a `SegTraQ` object.

# %%
sdata = sd.read_zarr("../../data/xenium_v1_data/sdata_xenium_crop.zarr/")
st = segtraq.SegTraQ(
    sdata,
    tables_centroid_x_key=None,
    tables_centroid_y_key=None,
    points_background_id=-1,  # "UNASSIGNED" for Xenium prime
)

sdata

# %% [markdown]
# As you can see, the `spatialdata`dataset contains cell and nuclear masks as `shapes`.
# It is important that you have a nuclear segmentation in your object,
# otherwise you will not be able to compute the metrics below.

# %% [markdown]
# ## Intersection over Union between cell and nucleus masks

# %% [markdown]
# First, we get the nucleus that overlaps most with each cell and compute the Intersection over Union (IoU)
# between cell and nuclear masks using the method `match_nuclei_to_cells()`.

# %%
results_df = st.rs.match_nuclei_to_cells()
results_df.head()

# %% [markdown]
# For each `cell_id`, we obtain the ID (`nucleus_id`) of the nucleus mask with the highest `IoU`.
# If a cell does not overlap with any nucleus, the function returns a missing value for `nucleus_id`.
# In addition to the `IoU`, we also report the fraction of the nucleus that overlaps with the cell.
# If the nucleus has an invalid geometry, `IoU` and `nucleus_fraction` are reported as `NA`.

# %% [markdown]
# Let's see what this looks like when we plot the IoU spatially.

# %%
# link annotations with cell boundaries
sdata.tables["table"].obs["region"] = "cell_boundaries"
sdata.set_table_annotates_spatialelement("table", region="cell_boundaries")

# plot
sdata.pl.render_shapes(
    element="cell_boundaries",
    color="iou",
    cmap="viridis",
    fill_alpha=0.5,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.render_shapes(
    element="nucleus_boundaries",
    fill_alpha=0.2,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.show(title="Overlay of nuclei and cell masks colored by IoU", colorbar=True)


# %% [markdown]
# We will quickly set up some helper functions to facilitate plotting.


# %%
# helper functions for plotting
def plot_histogram(
    df,
    column,
    bins=30,
    figsize=(6, 5),
    color="steelblue",
    edgecolor="black",
    show_median=True,
    median_kwargs=None,
    median_color="red",
    title=None,
    xlabel=None,
    ylabel="Count",
    ax=None,
):
    values = df[column].dropna()
    median_value = np.median(values)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.hist(values, bins=bins, color=color, edgecolor=edgecolor)

    if show_median:
        median_kwargs = median_kwargs or {}
        ax.axvline(
            median_value,
            linestyle="--",
            linewidth=2,
            color=median_color,
            label=f"Median = {median_value:.2f}",
            **median_kwargs,
        )
        ax.legend()

    ax.set_title(title or f"Distribution of {column}")
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel(ylabel)

    plt.show()


def plot_regression(
    df,
    x,
    y,
    figsize=(6, 6),
    dropna=True,
    ci=95,
    scatter_kws=None,
    line_kws=None,
    title=None,
    xlabel=None,
    ylabel=None,
    r2_loc=(0.05, 0.95),
    r2_fmt="{:.3f}",
    ax=None,
):
    data = df[[x, y]]
    if dropna:
        data = data.dropna()

    # Regression stats
    slope, intercept, r_value, p_value, std_err = linregress(data[x], data[y])
    r_squared = r_value**2

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    scatter_kws = scatter_kws or {"alpha": 0.6}
    line_kws = line_kws or {"color": "red"}

    sns.regplot(
        data=data,
        x=x,
        y=y,
        ci=ci,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        ax=ax,
    )

    # R² annotation
    ax.text(
        r2_loc[0],
        r2_loc[1],
        rf"$R^2 = {r2_fmt.format(r_squared)}$",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title or f"{y} vs. {x}")
    ax.grid(True)

    plt.show()


# %% [markdown]
# We can now investigate what the distribution of IoUs looks like.

# %%
plot_histogram(
    df=sdata["table"].obs,
    column="iou",
    xlabel="Intersection over Union (IoU)",
)

# %% [markdown]
# ## Expression similarity between cell and nucleus

# %% [markdown]
# Now that we have matched each cell with a nucleus, we can compare the transcript composition
# of the whole cell with that of its matched nucleus using `similarity_nucleus_cell()`.
# The two count profiles are transformed with PFlog1pPF and compared by cosine similarity.
#
# To reduce finite-count effects, SegTraQ generates a conditional permutation null by pooling the
# two profiles and randomly reallocating transcripts while preserving both region totals. The reported
# `similarity_nucleus_cell` is the observed cosine similarity minus the mean null similarity.
# Values near zero are close to the random-splitting expectation, whereas increasingly negative
# values indicate that cell and nucleus are less similar than expected. The lower-tail permutation
# p-value tests whether this reduction in similarity is stronger than expected under the null.

# %%
nucleus_cell_df = st.rs.similarity_nucleus_cell(n_permutations=200)

# %%
plot_histogram(
    df=sdata["table"].obs,
    column="similarity_nucleus_cell",
    title="Residual similarity between cell and nucleus",
    xlabel="Cell–nucleus similarity residual",
)

# %% [markdown]
# The residual score measures deviation from the finite-count null expectation. More negative
# values indicate that cell and nucleus are less similar than expected from random sampling alone.
# Such deviations can be compatible with transcript spillover or other assignment errors, although
# genuine subcellular RNA localization can also contribute.

# %%
plot_regression(
    df=sdata["table"].obs,
    x="iou",
    y="similarity_nucleus_cell",
    title="Cell–nucleus similarity residual vs. IoU",
    ylabel="Cell–nucleus similarity residual",
)

# %% [markdown]
# This plot can be used to assess whether the null-corrected similarity still depends strongly on
# the geometric overlap between the matched cell and nucleus.

# %% [markdown]
# The spatial plot below shows cell boundaries colored by the cell–nucleus similarity residual.
# More negative values indicate cells whose whole-cell and nuclear profiles are less similar than
# expected under the conditional random-splitting null.

# %%
# link annotations with cell boundaries
sdata.tables["table"].obs["region"] = "cell_boundaries"
sdata.set_table_annotates_spatialelement("table", region="cell_boundaries")

# plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

sdata.pl.render_shapes(
    element="nucleus_boundaries",
    fill_alpha=0.2,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.render_shapes(
    element="cell_boundaries",
    color="iou",
    cmap="viridis",
    fill_alpha=0.5,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.show(ax=ax[0], title="Overlay of nuclei and cell masks colored by IoU", colorbar=True)

sdata.pl.render_shapes(
    element="nucleus_boundaries",
    fill_alpha=0.2,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.render_shapes(
    element="cell_boundaries",
    color="similarity_nucleus_cell",
    cmap="viridis",
    fill_alpha=0.5,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.show(ax=ax[1], title="Cell boundaries colored by cell–nucleus similarity residual", colorbar=True)

# %% [markdown]
# ### Using the permutation p-value
#
# The similarity residual describes how far the observed cosine similarity lies from the mean
# permutation-null similarity. The lower-tail permutation p-value asks whether the observed similarity
# is unusually **low** under that null. Thus, cells with a negative residual and a small p-value show
# evidence of stronger regional disagreement than expected from finite transcript sampling alone.
#
# Because a p-value is computed for every cell, we control the false-discovery rate across cells.
# SciPy provides Benjamini–Hochberg correction through `scipy.stats.false_discovery_control`.
# Below, we adjust only non-missing p-values and mark cells with FDR < 0.05. Statistical significance
# should still be considered together with the residual magnitude rather than used as a standalone
# measure of biological importance. With `n_permutations=200`, the smallest attainable raw p-value is
# `1 / 201`; increase `n_permutations` when finer p-value resolution is required.

# %%
obs = sdata.tables["table"].obs
p_col = "similarity_nucleus_cell_p_value"
valid = obs[p_col].notna()

obs["similarity_nucleus_cell_q_value"] = np.nan
obs.loc[valid, "similarity_nucleus_cell_q_value"] = false_discovery_control(
    obs.loc[valid, p_col].to_numpy(),
    method="bh",
)

obs["similarity_nucleus_cell_significant"] = (
    obs["similarity_nucleus_cell_q_value"] < 0.05
)

obs.loc[
    valid,
    [
        "cell_id",
        "similarity_nucleus_cell",
        "similarity_nucleus_cell_p_value",
        "similarity_nucleus_cell_q_value",
        "similarity_nucleus_cell_significant",
    ],
].head()

# %% [markdown]
# The reported residual requires the permutation-null mean, so `n_permutations` must be at least 100.
# Increasing the number of permutations improves the precision of both the null mean and the p-value.

# %% [markdown]
# We can inspect cells with low IoU and strongly negative cell–nucleus similarity residuals to
# distinguish geometric mismatch from expression-profile disagreement.

# %%
obs = sdata["table"].obs
df = obs[["cell_id", "iou", "similarity_nucleus_cell"]].dropna()
df.loc[df["iou"] < 0.1].sort_values("similarity_nucleus_cell").head()

# %% [markdown]
# The cells at the top of this table combine weak geometric overlap with unusually low regional
# similarity. Plotting one example can help determine whether the score is consistent with
# transcript assignment outside the matched nuclear region.

# %%
cid = (
    df.loc[df["iou"] < 0.1]
    .sort_values("similarity_nucleus_cell")
    .iloc[0]["cell_id"]
)
cid


# %%
# helper function for plotting
def plot_cell_with_nucleus_and_transcripts(
    cid: float | int,
    title: str,
    pix_to_um_scale_factor: float = 0.2125,
    repositioned_transcripts: bool = False,
    padding=200,
    points_gene_key="feature_name",
    tables_cell_id_key="cell_id",
    points_cell_id_key="cell_id",
    genes=None,
    center_layer="nucleus_boundaries",
    outer_layer="cell_boundaries",
):
    # add annotation of this cell to .obs
    sdata["table"].obs["focal_cell"] = sdata["table"].obs.index == cid

    # compute x,y of cell and nucleus centroids in µm space
    centroid_x_cell_px = sdata["cell_boundaries"].loc[cid].geometry.centroid.x
    centroid_y_cell_px = sdata["cell_boundaries"].loc[cid].geometry.centroid.y
    centroid_x_cell = centroid_x_cell_px / pix_to_um_scale_factor
    centroid_y_cell = centroid_y_cell_px / pix_to_um_scale_factor

    nid = sdata["table"].obs.loc[sdata["table"].obs[tables_cell_id_key] == cid, "nucleus_id"]
    centroid_x_nucleus = sdata["nucleus_boundaries"].loc[nid].geometry.centroid.x / pix_to_um_scale_factor
    centroid_y_nucleus = sdata["nucleus_boundaries"].loc[nid].geometry.centroid.y / pix_to_um_scale_factor

    # add annotation of this cell to .points and build new `PointsModel``
    trans = sdata.points["transcripts"].compute()
    trans["focal_cell"] = "other_cells"
    trans.loc[trans[points_cell_id_key] == cid, "focal_cell"] = "focal_cell"
    trans["focal_cell"] = trans["focal_cell"].astype("category")
    if repositioned_transcripts:
        trans = trans.drop(columns=["x", "y", "z"])
        trans = trans.rename(columns={"repositioned_x": "x", "repositioned_y": "y", "repositioned_z": "z"})
    sdata.points["transcripts_2"] = sd.models.PointsModel.parse(trans)
    T = sd.transformations.get_transformation(sdata.points["transcripts"])
    sd.transformations.set_transformation(sdata.points["transcripts_2"], T)

    # zoom in for better visibility
    sdata_cropped = sdata.query.bounding_box(
        axes=["x", "y"],
        min_coordinate=[centroid_x_cell - padding, centroid_y_cell - padding],
        max_coordinate=[centroid_x_cell + padding, centroid_y_cell + padding],
        target_coordinate_system="global",
    )

    # plot cell and transcripts of that cell
    axes = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)[1]

    plot = sdata_cropped.pl.render_shapes(
        element=center_layer,
        fill_alpha=0.2,
        outline_alpha=1.0,
        outline_width=0.5,
        outline_color="black",
    ).pl.render_shapes(
        element=outer_layer,
        color="focal_cell",
        fill_alpha=0.1,
        outline_alpha=1.0,
        outline_width=0.5,
        outline_color="blue",
    )

    if genes is None:
        plot = plot.pl.render_points(
            "transcripts_2",
            color="focal_cell",
            alpha=0.1,
            groups=["focal_cell"],
            palette=["red"],
        )
    else:
        plot = plot.pl.render_points(
            "transcripts",
            color=points_gene_key,
            alpha=0.1,
            groups=genes,
            palette=["red"],
        )

    plot.pl.show(
        ax=axes,
        title=title,
        colorbar=True,
    )

    # landmark for cell and nucleus centroid
    axes.scatter([centroid_x_cell], [centroid_y_cell], marker="+", s=400, c="black", linewidths=2, zorder=10, alpha=0.4)
    axes.scatter([centroid_x_nucleus], [centroid_y_nucleus], marker="+", s=400, c="black", linewidths=2, zorder=10)


# %%
plot_cell_with_nucleus_and_transcripts(cid, title="Cell with low cell–nucleus similarity residual and low IoU")

# %% [markdown]
# Comparing the nucleus with the whole cell is not fully specific because the nuclear transcripts
# are also contained in the whole-cell profile. We therefore additionally compare transcripts in
# the matched nuclear region with transcripts in the remaining cell region (referred to here as the
# cytoplasm; for some segmentation methods this may also include reassigned transcripts outside the
# original morphological boundary).

# %% [markdown]
# ## Similarity between nucleus and cytoplasm
#
# `similarity_nucleus_cytoplasm()` compares nuclear and cytoplasmic gene-count profiles using the
# same PFlog1pPF cosine-similarity residual and conditional permutation null. The metric is defined
# only for cells with a matched nucleus and sufficient transcripts and genes in both regions
# (`min_transcripts`, `min_genes`).
#
# Residuals near zero are close to the null expectation, while negative values indicate that nucleus
# and cytoplasm are less similar than expected. Such deviations can reflect assignment errors or
# contamination, but may also arise from genuine subcellular RNA localization.

# %%
nuc_cyto_df = st.rs.similarity_nucleus_cytoplasm(n_permutations=200)
nuc_cyto_df.head()

# %% [markdown]
# The histogram below shows the distribution of nucleus–cytoplasm similarity residuals.

# %%
plot_histogram(
    df=sdata["table"].obs,
    column="similarity_nucleus_cytoplasm",
    xlabel="Nucleus–cytoplasm similarity residual",
)

# %% [markdown]
# The scatter plot below shows whether the null-corrected nucleus–cytoplasm similarity depends on
# the geometric overlap between the matched nucleus and cell.

# %%
plot_regression(
    df=sdata["table"].obs,
    x="iou",
    y="similarity_nucleus_cytoplasm",
    title="Nucleus–cytoplasm similarity residual vs. IoU",
    ylabel="Nucleus–cytoplasm similarity residual",
)

# %% [markdown]
# The spatial plots below compare geometric overlap with the two null-corrected regional similarity scores.

# %%
# link annotations with cell boundaries
sdata.tables["table"].obs["region"] = "cell_boundaries"
sdata.set_table_annotates_spatialelement("table", region="cell_boundaries")

axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)[1].flatten()

sdata.pl.render_shapes(
    element="nucleus_boundaries",
    fill_alpha=0.2,
    outline_width=0.5,
    outline_alpha=1.0,
    outline_color="black",
).pl.render_shapes(
    element="cell_boundaries",
    color="iou",
    cmap="viridis",
    fill_alpha=0.5,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.show(
    ax=axes[0],
    title="Intersection over Union (IoU)",
    colorbar=True,
    figsize=(6, 6),
)

sdata.pl.render_shapes(
    element="nucleus_boundaries",
    fill_alpha=0.2,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.render_shapes(
    element="cell_boundaries",
    color="similarity_nucleus_cell",
    cmap="viridis",
    fill_alpha=0.5,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.show(
    ax=axes[1],
    title="Nucleus vs. whole cell",
    colorbar=True,
    figsize=(6, 6),
)

sdata.pl.render_shapes(
    element="nucleus_boundaries",
    fill_alpha=0.2,
    outline_width=0.5,
    outline_alpha=1.0,
    outline_color="black",
).pl.render_shapes(
    element="cell_boundaries",
    color="similarity_nucleus_cytoplasm",
    cmap="viridis",
    fill_alpha=0.5,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.show(
    ax=axes[2],
    title="Nucleus vs. cytoplasm",
    colorbar=True,
    figsize=(6, 6),
)

# %% [markdown]
# Strongly negative regional similarity residuals despite high `IoU` illustrate that geometric agreement does not necessarily imply similar transcript composition.

# %% [markdown]
# ## Similarity between the cell center and border
#
# `similarity_center_border()` compares gene composition in the cell interior (“center”)
# and outer ring (“border”) using the PFlog1pPF cosine-similarity residual.
#
# Specifically, it:
#
# 1. Computes an equivalent radius for each cell and defines two erosion distances.
# 2. Constructs:
# - Border: outer ring of the cell
# - Center: inner eroded region
# - A buffer region between them is ignored
# 3. Assigns transcripts to center and border regions.
# 4. Builds center and border gene-count profiles.
# 5. Computes cosine similarity, subtracts the mean conditional-null similarity, and reports a lower-tail permutation p-value.

# %%
center_border_df = st.rs.similarity_center_border()

# %%
center_border_df.head()

# %% [markdown]
# The histogram below shows the distribution of the center–border similarity residual.

# %%
plot_histogram(
    df=sdata["table"].obs,
    column="similarity_center_border",
    xlabel="Center–border similarity residual",
)

# %% [markdown]
# The spatial plot below shows the spatial distribution of the computed similarity residual.

# %%
# link annotations with cell boundaries
sdata.tables["table"].obs["region"] = "cell_boundaries"
sdata.set_table_annotates_spatialelement("table", region="cell_boundaries")

sdata.pl.render_shapes(
    element="cell_centers",
    fill_alpha=0.2,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.render_shapes(
    element="cell_borders",
    fill_alpha=0.2,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.render_shapes(
    element="cell_boundaries",
    color="similarity_center_border",
    cmap="viridis",
    fill_alpha=0.5,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.show(
    title="Similarity residual: center vs. border",
    colorbar=True,
    figsize=(6, 6),
)

# %% [markdown]
# ## Similarity between the border and neighborhood
# `similarity_border_neighborhood()` compares gene composition in the cell border with that of
# its surrounding neighborhood using the same PFlog1pPF cosine-similarity residual.
#
# Specifically, it:
# 1. Defines the border region as the outer ring of each cell (with an inner buffer gap).
# 2. Identifies neighboring cells based on a distance threshold relative to cell size.
# 3. Aggregates transcripts from neighboring cells to obtain a neighborhood expression profile.
# 4. Builds border and neighborhood gene-count profiles.
# 5. Computes cosine similarity, subtracts the mean conditional-null similarity, and reports a lower-tail permutation p-value.

# %%
border_nh_df = st.rs.similarity_border_neighborhood()

# %%
border_nh_df.head()

# %% [markdown]
# The histogram below shows the distribution of the border–neighborhood similarity residual.

# %%
plot_histogram(
    df=sdata["table"].obs,
    column="similarity_border_neighborhood",
    xlabel="Border–neighborhood similarity residual",
)

# %% [markdown]
# The spatial plot below shows the spatial distribution of the computed similarity residual.

# %%
# link annotations with cell boundaries
sdata.tables["table"].obs["region"] = "cell_boundaries"
sdata.set_table_annotates_spatialelement("table", region="cell_boundaries")

sdata.pl.render_shapes(
    element="cell_centers",
    fill_alpha=0.2,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.render_shapes(
    element="cell_borders",
    fill_alpha=0.2,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.render_shapes(
    element="cell_boundaries",
    color="similarity_border_neighborhood",
    cmap="viridis",
    fill_alpha=0.5,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.show(
    title="Similarity residual: border vs. neighborhood",
    colorbar=True,
    figsize=(6, 6),
)

# %% [markdown]
# The border can resemble both the center and the neighborhood. Because these are residual similarities
# relative to pair-specific permutation nulls, their direct difference is only a descriptive contrast:
#
# $$
# \Delta = R_{\mathrm{border,neighborhood}} - R_{\mathrm{center,border}}.
# $$
#
# Positive values mean that border–neighborhood similarity is higher relative to its own null than
# center–border similarity is relative to its null. Because the two residuals have different pooled
# profiles and null distributions, this contrast should not be interpreted as a formal contamination test.

# %%
obs = sdata.tables["table"].obs
obs["border_neighborhood_similarity_contrast"] = (
    obs["similarity_border_neighborhood"] - obs["similarity_center_border"]
)

# %%
# link annotations with cell boundaries
obs["region"] = "cell_boundaries"
sdata.set_table_annotates_spatialelement("table", region="cell_boundaries")

axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)[1].flatten()

sdata.pl.render_shapes(element="cell_boundaries", color="similarity_center_border").pl.show(
    ax=axes[0], title="Center–border similarity residual"
)
sdata.pl.render_shapes(element="cell_boundaries", color="similarity_border_neighborhood").pl.show(
    ax=axes[1], title="Border–neighborhood similarity residual"
)
sdata.pl.render_shapes(element="cell_boundaries", color="border_neighborhood_similarity_contrast").pl.show(
    ax=axes[2], title="Border–neighborhood minus center–border"
)

# %% [markdown]
# ## Border admixture score
# Comparing the two regional similarity residuals can show how the border relates to center and
# neighborhood relative to their respective null expectations, but it does not directly estimate a
# neighborhood contribution. The **border admixture score** instead explicitly models the border as a mixture of
# center and neighborhood expression.
#
# Specifically, the function `border_admixture_score()`:
#
# 1. Computes gene expression profiles for center, border, and neighborhood regions.
# 2. Converts counts to gene proportions (with a small pseudocount).
# 3. Models the border profile as a mixture:
#
#   $$
#   p_{\text{border}} \approx (1 - \alpha)\, p_{\text{center}} + \alpha\, p_{\text{neighborhood}}
#   $$
#
# 4. Estimates the mixture weight $\alpha$ using least squares.
# 5. Computes how much better this mixture explains the border compared to the center alone (`border_admixture_score`).
# 6. Estimates confidence intervals via bootstrap resampling.
#
# The resulting score reflects how strongly the border resembles the neighborhood beyond what
# is expected from the center alone.

# %%
st.rs.border_admixture_score(n_jobs=-1)

# %% [markdown]
# The histogram below shows the distribution of the border_admixture_score.

# %%
plot_histogram(
    df=sdata["table"].obs,
    column="border_admixture_score",
    xlabel="Border admixture score",
)

# %% [markdown]
# The confidence interval reflects the uncertainty in the border admixture score due to
# limited and noisy transcript counts, estimated via bootstrap resampling.
# It should be considered to distinguish robust signals from effects that could arise by chance,
# especially in sparse data.
#
# A border_admixture_score > 0 indicates that including the neighborhood improves the fit to the
# border compared to using the center alone. A score of 1 means that the border expression is
# perfectly explained by a mixture of center and neighborhood.
#
# To identify potentially contaminated cells, we use a threshold of 0.25 and require that
# the lower bound of the confidence interval exceeds this threshold,
# ensuring that only cells with a robust and consistent neighborhood contribution are selected.

# %%
obs = sdata.tables["table"].obs

obs["border_admixture_score_confident_binary"] = obs["border_admixture_score_ci_low"] > 0.25

obs["border_admixture_score_confident"] = obs["border_admixture_score"]

obs.loc[~obs["border_admixture_score_confident_binary"], "border_admixture_score_confident"] = np.nan

# %%
# link annotations with cell boundaries
sdata.tables["table"].obs["region"] = "cell_boundaries"
sdata.set_table_annotates_spatialelement("table", region="cell_boundaries")

axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)[1].flatten()

sdata.pl.render_shapes(element="cell_boundaries", color="border_admixture_score").pl.show(
    ax=axes[0], title="Border admixture score (all cells)"
)

sdata.pl.render_shapes(element="cell_boundaries", color="border_admixture_score_confident_binary").pl.show(
    ax=axes[1], title="Cells with significant neighborhood contribution (CI > 0.25)"
)

sdata.pl.render_shapes(element="cell_boundaries", color="border_admixture_score_confident").pl.show(
    ax=axes[2], title="Border admixture score (confidence-filtered)"
)

# %% [markdown]
# ## Session Info

# %%
print(sd.__version__)  # spatialdata
print(spatialdata_plot.__version__)

# %%
