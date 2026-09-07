import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import spatialdata as sd

import segtraq as st
from segtraq.utils import _filter_control_and_low_quality_transcripts

st.settings.n_jobs = -1

EXPECTED_SNAPSHOTS = {
    "baseline": None,
    "region_similarity": None,
    "plotting": None,
    "point_statistics": None,
    "clustering": None,
    "supervised": None,
    "volume": None,
}


def _pick_present_gene(sdata, points_key="transcripts", points_gene_key="feature_name", n=2000):
    sample = sdata.points[points_key].head(n)
    if hasattr(sample, "compute"):
        sample = sample.compute()
    return sample[points_gene_key].dropna().astype(str).iloc[0]


def _normalize_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.floating, float)):
        return round(float(value), 8)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return str(value)


def _df_snapshot(df: pd.DataFrame):
    records = []
    for record in df.to_dict(orient="records"):
        records.append({str(k): _normalize_value(v) for k, v in record.items()})
    records.sort(key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")))
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "columns": sorted([str(c) for c in df.columns]),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def _scalar_snapshot(value):
    return _normalize_value(value)


def _markers_snapshot(markers):
    normalized = {}
    for cell_type, marker_dict in markers.items():
        normalized[str(cell_type)] = {
            "positive": sorted([str(g) for g in marker_dict["positive"]]),
            "negative": sorted([str(g) for g in marker_dict["negative"]]),
        }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return {
        "cell_types": sorted(list(normalized.keys())),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def _axes_snapshot(axes):
    flat_axes = []
    for item in axes:
        if isinstance(item, list):
            flat_axes.extend(item)
        else:
            flat_axes.append(item)

    snapshots = []
    for ax in flat_axes:
        lines = []
        for line in ax.lines:
            x = [_normalize_value(v) for v in line.get_xdata()]
            y = [_normalize_value(v) for v in line.get_ydata()]
            lines.append({"x": x, "y": y})

        snapshots.append(
            {
                "title": ax.get_title(),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "xlim": [_normalize_value(v) for v in ax.get_xlim()],
                "ylim": [_normalize_value(v) for v in ax.get_ylim()],
                "line_count": len(ax.lines),
                "sha256": hashlib.sha256(json.dumps(lines, sort_keys=True).encode()).hexdigest(),
            }
        )

    snapshots.sort(key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")))
    return snapshots


def _assert_snapshot(name, actual):
    expected = EXPECTED_SNAPSHOTS[name]
    if expected is None:
        pytest.fail(f"Populate snapshot for '{name}': {json.dumps(actual, sort_keys=True)}")
    assert actual == expected


def test_snapshot_baseline_methods(sdata_new):
    actual = {
        "num_cells": _scalar_snapshot(st.bl.num_cells(sdata_new, inplace=False)),
        "num_transcripts": _scalar_snapshot(st.bl.num_transcripts(sdata_new, inplace=False)),
        "num_genes": _scalar_snapshot(st.bl.num_genes(sdata_new, inplace=False)),
        "perc_unassigned_transcripts": _scalar_snapshot(st.bl.perc_unassigned_transcripts(sdata_new, inplace=False)),
        "perc_unassigned_transcripts_per_gene": _df_snapshot(
            st.bl.perc_unassigned_transcripts_per_gene(sdata_new, inplace=False)
        ),
        "transcripts_per_cell": _df_snapshot(st.bl.transcripts_per_cell(sdata_new, inplace=False)),
        "genes_per_cell": _df_snapshot(st.bl.genes_per_cell(sdata_new, inplace=False)),
        "mean_transcripts_per_gene_per_cell": _df_snapshot(
            st.bl.mean_transcripts_per_gene_per_cell(sdata_new, inplace=False)
        ),
        "morphological_features": _df_snapshot(st.bl.morphological_features(sdata_new, inplace=False)),
    }
    _assert_snapshot("baseline", actual)


def test_snapshot_region_similarity_methods(segtraq_obj, sdata_new):
    sdata_filtered = _filter_control_and_low_quality_transcripts(sd.deepcopy(sdata_new))
    actual = {
        "match_nuclei_to_cells": _df_snapshot(
            st.rs.match_nuclei_to_cells(sd.deepcopy(sdata_new), inplace=False, n_jobs=8)
        ),
        "similarity_nucleus_cell": _df_snapshot(st.rs.similarity_nucleus_cell(sdata_filtered, inplace=False, n_jobs=8)),
        "similarity_nucleus_cytoplasm": _df_snapshot(
            st.rs.similarity_nucleus_cytoplasm(sd.deepcopy(sdata_new), inplace=False)
        ),
        "border_admixture_score": _df_snapshot(st.rs.border_admixture_score(sd.deepcopy(sdata_new), inplace=False)),
    }

    segtraq_obj.run_region_similarity(inplace=True)
    run_cols = ["iou", "similarity_nucleus_cell", "similarity_nucleus_cytoplasm", "border_admixture_score"]
    actual["run_region_similarity"] = _df_snapshot(segtraq_obj.sdata.tables["table"].obs[run_cols].reset_index())

    _assert_snapshot("region_similarity", actual)


@pytest.mark.filterwarnings("ignore:.*No artists with labels found to put in legend.*:UserWarning")
def test_snapshot_plotting_methods(segtraq_obj, sdata_new):
    st_dict = {"test1": segtraq_obj, "test2": segtraq_obj}
    actual = {
        "celltype_proportions": _df_snapshot(st.pl.celltype_proportions(st_dict, celltype_col="transferred_cell_type")),
        "boxplot": _df_snapshot(
            st.pl.boxplot(st_dict, celltype_col="transferred_cell_type", value_key="transcript_count")
        ),
        "boxplot_combined": _df_snapshot(
            st.pl.boxplot_combined(st_dict, celltype_col="transferred_cell_type", value_key="transcript_count")
        ),
    }

    tx_axes = st.pl.transcript_distribution_across_space(sd.deepcopy(sdata_new))
    actual["transcript_distribution_across_space"] = _axes_snapshot(tx_axes)

    feat_axes = st.pl.feature_distribution_across_space(sd.deepcopy(sdata_new), features=["transcript_counts"])
    actual["feature_distribution_across_space"] = _axes_snapshot(feat_axes)

    _assert_snapshot("plotting", actual)


def test_snapshot_point_statistics_methods(sdata_labeled, sdata_new):
    gene = _pick_present_gene(sdata_new)
    actual = {
        "gene": gene,
        "percentage_transcripts_in_compartments": _df_snapshot(
            st.ps.percentage_transcripts_in_compartments(sd.deepcopy(sdata_labeled), genes=gene, inplace=False)
        ),
        "distance_to_centroid": _df_snapshot(
            st.ps.distance_to_centroid(sd.deepcopy(sdata_new), genes=gene, centroid_region="cell", inplace=False)
        ),
        "distance_to_membrane": _df_snapshot(
            st.ps.distance_to_membrane(
                sd.deepcopy(sdata_new),
                genes=gene,
                membrane_region="cell",
                restrict_to_within_boundary=False,
                signed=True,
                inplace=False,
            )
        ),
        "membrane_distance_skewness": _df_snapshot(
            st.ps.membrane_distance_skewness(sd.deepcopy(sdata_new), genes=gene, min_transcripts=5, inplace=False)
        ),
    }
    _assert_snapshot("point_statistics", actual)


def test_snapshot_clustering_stability_methods(sdata_new):
    actual = {
        "cluster_connectedness": _scalar_snapshot(
            st.cs.cluster_connectedness(
                sd.deepcopy(sdata_new), resolution=1.0, key_prefix="leiden_subset", random_state=42
            )
        ),
        "silhouette_score": _scalar_snapshot(
            st.cs.silhouette_score(sd.deepcopy(sdata_new), resolution=1.0, key_prefix="leiden_subset", random_state=42)
        ),
        "purity": _scalar_snapshot(st.cs.purity(sd.deepcopy(sdata_new), resolution=1.0, key_prefix="leiden_subset")),
        "adjusted_rand_index": _scalar_snapshot(
            st.cs.adjusted_rand_index(sd.deepcopy(sdata_new), resolution=1.0, key_prefix="leiden_subset")
        ),
    }
    _assert_snapshot("clustering", actual)


def test_snapshot_supervised_methods(adata_ref, markers, sdata_3D_labeled, sdata_labeled):
    per_cell_df, mat_df, str_df, n_eval_df = st.sp.neighbor_contamination(
        sdata=sd.deepcopy(sdata_labeled),
        cell_type_key="transferred_cell_type",
        markers=markers,
        tables_key="table",
        tables_cell_id_key="cell_id",
        neighbors_key="spatial_connectivities",
        inplace=False,
    )

    actual = {
        "markers_from_reference": _markers_snapshot(
            st.markers_from_reference(adata_ref.copy(), ref_cell_type="celltype", ref_raw_counts_layer="raw")
        ),
        "mutually_exclusive_coexpression_rate": _df_snapshot(
            st.sp.mutually_exclusive_coexpression_rate(
                sdata=sd.deepcopy(sdata_3D_labeled),
                markers=markers,
                tables_key="table",
                inplace=False,
            )
        ),
        "neighbor_contamination_per_cell": _df_snapshot(per_cell_df),
        "neighbor_contamination_matrix": _df_snapshot(mat_df),
        "neighbor_contamination_strength_matrix": _df_snapshot(str_df),
        "neighbor_contamination_n_evaluable_matrix": _df_snapshot(n_eval_df),
        "marker_purity": _df_snapshot(
            st.sp.marker_purity(
                sdata=sd.deepcopy(sdata_labeled),
                cell_type_key="transferred_cell_type",
                markers=markers,
                neighbors_key="spatial_connectivities",
                inplace=False,
            )
        ),
    }
    _assert_snapshot("supervised", actual)


def test_snapshot_volume_methods(sdata_3D_labeled, sdata_new):
    shapes_key_list = ["cell_boundaries_z0", "cell_boundaries_z1", "cell_boundaries_z2", "cell_boundaries_z3"]
    actual = {
        "vertical_signal_integrity_per_cell": _df_snapshot(
            st.vl.vertical_signal_integrity_per_cell(
                sd.deepcopy(sdata_new),
                ovrlpy_init_kwargs={"n_components": 10},
                inplace=False,
            )
        ),
        "similarity_top_bottom": _df_snapshot(st.vl.similarity_top_bottom(sd.deepcopy(sdata_new), inplace=False)),
        "fraction_heterotypic_overlap": _df_snapshot(
            st.vl.fraction_heterotypic_overlap(
                sd.deepcopy(sdata_3D_labeled),
                tables_cell_id_key="cell",
                shapes_cell_id_key="cell",
                shapes_key_list=shapes_key_list,
                inplace=False,
            )
        ),
    }
    _assert_snapshot("volume", actual)
