from __future__ import annotations

from collections.abc import Mapping

import anndata as ad
import numpy as np
import pandas as pd


def _fill_celltype(s: pd.Series, missing_label: str = "None") -> pd.Series:
    """
    Fill missing cell type labels in a Series. Required for spatialdata-plot to work.
    """
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


def build_umap_df(
    method_to_adata: dict[str, ad.AnnData],
    feature_col: str,
    umap_key: str = "X_umap",
) -> pd.DataFrame:
    rows = []
    for method, adata in method_to_adata.items():
        if umap_key not in adata.obsm:
            raise KeyError(f"{method}: adata.obsm['{umap_key}'] not found.")
        umap = np.asarray(adata.obsm[umap_key])
        if umap.shape[1] != 2:
            raise ValueError(f"{method}: {umap_key} must be n_cells x 2.")

        feature = adata.obs[feature_col]
        rows.append(
            pd.DataFrame(
                {
                    "x": umap[:, 0],
                    "y": umap[:, 1],
                    feature_col: feature.values,
                    "Segmentation Method": method,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


# TODO: WILL BE DEPRECATED, REMOVE
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
            raise KeyError(f"{method}: adata.obs['{value_key}'] not found. Available keys: {list(adata.obs.columns)}")
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
