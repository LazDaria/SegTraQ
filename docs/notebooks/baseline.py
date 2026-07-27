# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: segtraq2_env
#     language: python
#     name: segtraq2_env
# ---

# %% [markdown]
# # Module: Baseline Metrics
#
# When assessing the quality of a segmentation, the first thing one usually looks at are summary metrics
# such as the number of segmented cells, number of transcripts/genes per cell,
# the percentage of unassigned transcripts, the transcript density, and a variety of morphological features.
#
# <center>
#  <img src='../_static/img/docs/baseline.png' width='90%' />
# </center>
#
# The `baseline` (`bl`) module contains several metrics that can help you to assess the quality of your segmentation.
# The methods all return their corresponding values or dataframes, and also write them into the spatialdata object.
#
# To follow along with this tutorial, you can download the data from [here](https://oc.embl.de/index.php/s/iGxVy8qtZnwHOju).

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import math

import matplotlib.pyplot as plt
import spatialdata as sd

import segtraq

sdata = sd.read_zarr("../../data/xenium_5K_data/proseg2.zarr")

# putting the spatialdata object into a SegTraQ constructor
# this has the advantage that we only need to set keywords like cell IDs or transcript IDs once
st = segtraq.SegTraQ(
    sdata,
    images_key=None,
    tables_area_key=None,
    points_background_id=0,
    tables_centroid_x_key="centroid_x",
    tables_centroid_y_key="centroid_y",
)

# %% [markdown]
# To get a first impression of the quality of the data, we can check how many cells there are in the data.
# We can also check how many transcripts were measured in total,
# how many genes they map to, and how many transcripts were not assigned to a cell.

# %%
st.bl.num_cells()

# %%
st.bl.num_transcripts()

# %%
st.bl.num_genes()

# %%
st.bl.perc_unassigned_transcripts()

# %% [markdown]
# Next, let's see how many transcripts were detected per cell.
# We can do this using the `transcripts_per_cell()` method.

# %%
transcripts_per_cell = st.bl.transcripts_per_cell()
transcripts_per_cell.head()

# %% [markdown]
# We can plot the median and distribution of this to see how the number of transcripts differs across cells.

# %%
# plotting the number of transcripts per cell
plt.figure(figsize=(10, 6))
plt.hist(transcripts_per_cell["transcript_count"], bins=100)

# adding a line for the median
plt.axvline(
    transcripts_per_cell["transcript_count"].median(),
    color="black",
    linestyle="dashed",
    linewidth=1,
)
plt.text(
    transcripts_per_cell["transcript_count"].median() + 5,
    50,
    f"Median: {transcripts_per_cell['transcript_count'].median():.2f}",
    color="black",
)

# adding labels and title
plt.xlabel("Number of Transcripts")
plt.ylabel("Count")
plt.title("Distribution of Transcripts per Cell")
plt.show()

# %% [markdown]
# We can also do the same thing for the number of genes per cell,
# since there are often multiple transcripts measured per gene.

# %%
genes_per_cell = st.bl.genes_per_cell()
genes_per_cell.head()

# %%
# plotting the number of genes per cell
plt.figure(figsize=(10, 6))
plt.hist(genes_per_cell["gene_count"], bins=100)

# adding a line for the median
plt.axvline(
    genes_per_cell["gene_count"].median(),
    color="black",
    linestyle="dashed",
    linewidth=1,
)
plt.text(
    genes_per_cell["gene_count"].median() + 5,
    50,
    f"Median: {genes_per_cell['gene_count'].median():.2f}",
    color="black",
)

# adding labels and title
plt.xlabel("Number of Genes")
plt.ylabel("Count")
plt.title("Distribution of Genes per Cell")
plt.show()

# %% [markdown]
# Next to the number of transcripts per cell, we can also investigate the transcript density,
# which is computed as the number of transcripts divided by the cell area.
# Note that the background does not appear in this data frame.

# %%
transcript_density = st.bl.transcript_density()
transcript_density.head()

# %%
x = transcript_density["transcript_density"].dropna()

p99 = x.quantile(0.99)
x_clip = x[x <= p99]

plt.figure(figsize=(10, 6))
plt.hist(x_clip, bins=100)

# median from full distribution
med = x.median()

plt.axvline(med, color="black", linestyle="dashed", linewidth=1)
plt.text(
    med + 0.05,
    plt.ylim()[1] * 0.9,
    f"Median: {med:.2f}",
    color="black",
)

# adding labels and title
plt.xlabel("Transcript Density (transcripts per area)")
plt.ylabel("Count")
plt.title("Distribution of Transcript Density per Cell")
plt.show()

# %% [markdown]
# We can also compute the mean number of transcripts per detected gene per cell,
# which is computed by averaging per-gene transcript counts across genes observed in each cell.
# Note that only detected genes are considered and background transcripts are excluded.

# %%
mean_transcripts_per_gene_per_cell = st.bl.mean_transcripts_per_gene_per_cell()
mean_transcripts_per_gene_per_cell.head()

# %%
x = mean_transcripts_per_gene_per_cell["mean_transcripts_per_gene"].dropna()

p99 = x.quantile(0.99)
x_clip = x[x <= p99]

plt.figure(figsize=(10, 6))
plt.hist(x_clip, bins=100)

# median from full distribution
med = x.median()

plt.axvline(med, color="black", linestyle="dashed", linewidth=1)
plt.text(
    med + 0.05,
    plt.ylim()[1] * 0.9,
    f"Median: {med:.2f}",
    color="black",
)

# adding labels and title
plt.xlabel("Transcript Density (transcripts per area)")
plt.ylabel("Count")
plt.title("Distribution of Transcript Density per Cell")
plt.show()

# %% [markdown]
# Finally, let's look at some morphological features,
# such as the cell area, circularity, elongation, etc.
# We can get those with the function `morphological_features()`.
# If you only want to compute certain features, you can select them with the
# `features_to_compute` argument. This can drastically reduce the runtime,
# as especially the features `elongation` and `eccentricity` can take a while to compute.

# %%
morphological_features = st.bl.morphological_features()
morphological_features.head()

# %% [markdown]
# Let's plot all of the distributions in one plot using subplots.

# %%
# exclude 'cell_id' from the features to plot
features = [f for f in morphological_features.columns if f != "cell_id"]

# define grid layout
num_features = len(features)
cols = 3
rows = math.ceil(num_features / cols)

# create subplots
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
axes = axes.flatten()

for i, feature in enumerate(features):
    ax = axes[i]
    data = morphological_features[feature]

    ax.hist(data, bins=100)
    median_val = data.median()
    ax.axvline(median_val, color="black", linestyle="dashed", linewidth=1)
    ax.text(
        median_val + 0.05,
        ax.get_ylim()[1] * 0.9,
        f"Median: {median_val:.2f}",
        color="black",
        fontsize=8,
    )

    ax.set_title(feature, fontsize=10)
    ax.set_xlabel("Value", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)

# hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
plt.show()

# %% [markdown]
# Instead of measures per cell, we can also compute some metrics per gene.
# For example, we can see the percentage of how often a gene was not assigned to any cell.
# This can help to detect potential biases in our segmentation.

# %%
perc_unassigned_transcripts_per_gene = st.bl.perc_unassigned_transcripts_per_gene()
perc_unassigned_transcripts_per_gene.sort_values(by="perc_unassigned", ascending=False).head()

# %% [markdown]
# In our example, most transcripts were assigned to a cell.
# However, if you detected that a large number of transcripts were unassigned,
# you could follow up with a gene set enrichment analysis (GSEA) to look for specific biases.

# %% [markdown]
# Finally, we can check the anndata object to verify that all of our metrics are stored in there.

# %%
sdata.tables["table"]

# %% [markdown]
# Alternatively, all `bl` metrics can be computed in one run via `run_baseline`.

# %%
st.run_baseline()
sdata.tables["table"].obs.head()

# %%
sdata.tables["table"].var.head()

# %% [markdown]
# ## Session Info

# %%
print(sd.__version__)  # spatialdata
