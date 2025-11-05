from pathlib import Path
from typing import Any, Optional
from anndata import AnnData

import numpy as np
import scanpy as sc
import warnings

from . import bl, cs, nc, pl, sp
from .utils import run_label_transfer as _run_label_transfer

from .utils import validate_spatialdata


class SegTraQ:
    def __init__(
        self,
        sdata,
        images_key: str | None = "morphology_focus",
        tables_key: str = "table",
        tables_cell_id_key: str = "cell_id",
        tables_area_volume_key: str | None = "cell_area",
        points_key: str = "transcripts",
        points_cell_id_key: str = "cell_id",
        points_background_id: str | int = "UNASSIGNED",
        points_x_key: str = "x",
        points_y_key: str = "y",
        points_z_key: str | None = "z",
        points_gene_key: str = "feature_name",
        shapes_key: str = "cell_boundaries",
        shapes_cell_id_key: str | None = "cell_id",
        nucleus_shapes_key: str | None = "nucleus_boundaries",
        nucleus_shapes_cell_id_key: str | None = "cell_id",
        labels_key: str | None = "cell_labels",
        labels_data_key: str | None = "scale0/image",
        labels_to_cell_id_key: str | None = "cell_labels",
        cell_type_key: str | None = "transferred_cell_type",
    ):       

        """
        Initialize a SegTraQ object, the core interface for computing SegTraQ metrics.
        Defaults target 10x Genomics Xenium; override keys for other technologies.

        Parameters
        ----------
        sdata : SpatialData
            A `SpatialData` object containing segmented and transcript-assigned spatial
            transcriptomics data (images, tables, points, shapes and optional labels).

        images_key : str or None, optional, default="morphology_focus"
            Key in `sdata.images` for a nuclear or morphology image (e.g., DAPI).
            Used for visualization or to derive a nucleus mask via `segtraq.run_cellpose`
            when using the nuclear correlation module (`segtraq.nc`). If `None`, no image
            is expected.

        tables_key : str, default="table"
            Key in `sdata.tables` for the cell-level metadata table. Gene names in
            `sdata.tables[tables_key].var.index` should match the gene field in
            `sdata.points[points_key]` (see `points_gene_key`).

        tables_cell_id_key : str, default="cell_id"
            Column in the cell table uniquely identifying each cell.

        tables_area_volume_key : str or None, optional, default="cell_area"
            Column in the cell table with cell area (2D) or volume (3D/quasi-3D).
            If `None`, area/volume-based metrics will be computed via
            `segtraq.bl.morphological_features`.

        points_key : str, default="transcripts"
            Key in `sdata.points` for spot/transcript-level data.

        points_cell_id_key : str, default="cell_id"
            Column in the points table linking each transcript/spot to a cell.

        points_background_id : str or int, default="UNASSIGNED"
            Identifier for transcripts not assigned to any cell (background).

        points_x_key : str, default="x"
            Column for the x-coordinate of each transcript/spot.

        points_y_key : str, default="y"
            Column for the y-coordinate of each transcript/spot.

        points_z_key : str or None, optional, default="z"
            Column for the z-coordinate (3D data). If `None`, data are treated as 2D.

        points_gene_key : str, default="feature_name"
            Column specifying the gene/feature name for each transcript/spot.

        shapes_key : str, default="cell_boundaries"
            Key in `sdata.shapes` for cell boundary polygons.

        shapes_cell_id_key : str or None, optional, default="cell_id"
            Column in the cell-boundary shapes linking polygons to cell IDs.
            If `None`, the shape index is used as the cell ID.

        nucleus_shapes_key : str or None, optional, default="nucleus_boundaries"
            Key in `sdata.shapes` for nucleus boundary polygons, if available.
            If None, a nucleus mask can be obtained via `segtraq.run_cellpose`.

        nucleus_shapes_cell_id_key : str or None, optional, default="cell_id"
            Column linking nucleus polygons to cell IDs. If `None` but
            `nucleus_shapes_key` is provided, the shape index is used as the cell ID.

        labels_key : str or None, optional, default="cell_labels"
            Key in `sdata.labels` for a labeled segmentation mask, if available.

        labels_data_key : str or None, optional, default="scale0/image"
            Key for accessing data in `sdata.labels` if they are stored as a DataTree. Default is None.

        labels_to_cell_id_key : str or None, optional, default="cell_labels"
            Column in `sdata.tables[tables_key]` mapping segmentation label IDs
            (from `labels_key`) to cell IDs.

        cell_type_key : str or None, optional, default="transferred_cell_type"
            Column in the cell table containing cell-type annotations, if available.
            Note: running `segtraq.run_label_transfer()` will overwrite this column.
        """

        # Validate spatialdata object
        validate_spatialdata(
            sdata,
            images_key=images_key,
            tables_key=tables_key,
            tables_cell_id_key=tables_cell_id_key,
            tables_area_volume_key=tables_area_volume_key,
            points_key=points_key,
            points_cell_id_key=points_cell_id_key,
            points_background_id=points_background_id,
            points_x_key=points_x_key,
            points_y_key=points_y_key,
            points_z_key=points_z_key,
            points_gene_key=points_gene_key,
            shapes_key=shapes_key,
            shapes_cell_id_key=shapes_cell_id_key,
            nucleus_shapes_key=nucleus_shapes_key,
            nucleus_shapes_cell_id_key=nucleus_shapes_cell_id_key,
            labels_key=labels_key,
            labels_to_cell_id_key=labels_to_cell_id_key,
            cell_type_key=cell_type_key,
            labels_data_key=labels_data_key,
        )

        self.sdata = sdata

        self.images_key = images_key

        self.tables_key = tables_key
        self.tables_cell_id_key = tables_cell_id_key
        self.tables_area_volume_key = tables_area_volume_key

        self.points_key = points_key
        self.points_cell_id_key = points_cell_id_key
        self.points_background_id = points_background_id
        self.points_x_key = points_x_key
        self.points_y_key = points_y_key
        self.points_z_key = points_z_key
        self.points_gene_key = points_gene_key

        self.shapes_key = shapes_key
        self.shapes_cell_id_key = shapes_cell_id_key
        self.nucleus_shapes_key = nucleus_shapes_key
        self.nucleus_shapes_cell_id_key = nucleus_shapes_cell_id_key

        self.labels_key = labels_key
        self.labels_data_key = labels_data_key
        self.labels_to_cell_id_key = labels_to_cell_id_key

        self.cell_type_key = cell_type_key

        self.bl = _BLFacade(self)

    def run_baseline(self, inplace: bool = True):
        """
        Compute baseline SegTraQ metrics (via the bound BL facade) and optionally
        merge them into the cell table.

        Metrics
        -------
        - genes_per_cell
        - transcripts_per_cell
        - transcript_density (only if `tables_area_volume_key` is set)
        - morphological features per cell
        - global count metrics (num_cells, num_genes, num_transcripts, perc_unassigned_transcripts)

        Parameters
        ----------
        inplace : bool, default=True
            If True, metrics are merged into `sdata.tables[tables_key].obs` and
            `sdata.tables[tables_key].uns.
            If False, returns the computed objects.

        Returns
        -------
        None or dict
            - If `inplace=True`: returns None after writing to `sdata`.
            - If `inplace=False`: returns a dict with keys:
            `summary`, `genes_per_cell`, `transcripts_per_cell`, and optionally `transcript_density`.
        """
        tbl = self.sdata.tables[self.tables_key]

        gpc = self.bl.genes_per_cell(inplace=inplace).set_index(self.tables_cell_id_key)
        tpc = self.bl.transcripts_per_cell(inplace=inplace).set_index(self.tables_cell_id_key)
        mrp = self.bl.morphological_features(inplace=inplace).set_index(self.shapes_cell_id_key)

        dens_raw = self.bl.transcript_density(inplace=inplace)
        dens = None if dens_raw is None else dens_raw.set_index(self.tables_cell_id_key)

        summary = dict(
            num_cells=self.bl.num_cells(inplace=inplace),
            num_genes=self.bl.num_genes(inplace=inplace),
            num_transcripts=self.bl.num_transcripts(inplace=inplace),
            perc_unassigned_transcripts=self.bl.perc_unassigned_transcripts(inplace=inplace),
        )

        if inplace:
            return None
        else:
            out = {
                "summary": summary,
                "genes_per_cell": gpc,
                "transcripts_per_cell": tpc,
                "morphological_feautres": mrp,
            }
            if dens is not None:
                out["transcript_density"] = dens
            return out
        
    def run_label_transfer(self, 
                           adata_ref = AnnData, 
                           tx_min: float = 10.0,
                           tx_max: float = 2000.0,
                           gn_min: float = 5.0,
                           gn_max: float = np.inf,
                           ref_cell_type: str = "cell_type",
                           ref_ensemble_key: str | None = None,
                           query_ensemble_key: str | None =  "gene_ids",
                           inplace: bool = True):
        """
        Transfer cell-type labels from a reference AnnData to the current SpatialData table.
        Cells are optionally filtered by per-cell transcript and gene counts before transfer.

        Parameters
        ----------
        adata_ref : AnnData
            Reference AnnData with cell-type annotations in `.obs[self.ref_cell_type]`.
        tx_min, tx_max : float, default=(10.0, 2000.0)
            Inclusive lower and upper bounds for per-cell transcript count filtering.
        gn_min, gn_max : float, default=(5.0, inf)
            Inclusive lower and upper bounds for per-cell gene count filtering.
        ref_cell_type: str, default="cell_type"
            Column name of cell-type annotations in `adata_ref.obs[ref_cell_type]`.
        ref_ensemble_key: str or None, default=None
            Column name in `adata_ref.var` that contains unique gene/ensemble IDs. If None, `adata_ref.var_names` will be used.
        query_ensemble_key: str or None, default="gene_ids"
            Column name in `self.sdata.tables[self.tables_key].var` that contains unique gene/ensemble IDs. If None, `self.sdata.tables[self.tables_key].var_names` will be used.
        inplace : bool, default=True
            If True, writes labels/scores into `sdata.tables[tables_key].obs` and returns None.
            If False, returns a DataFrame with the assignment and scores without writing.

        Returns
        -------
        None or pd.DataFrame
            None when `inplace=True`; otherwise a DataFrame of assignments.
        """

        # Delegate to utility (aliased to avoid name confusion)
        result = _run_label_transfer(
            sdata=self.sdata,
            adata_ref=adata_ref,
            ref_cell_type_key=ref_cell_type,
            tables_key=self.tables_key,
            tx_min=tx_min,
            tx_max=tx_max,
            gn_min=gn_min,
            gn_max=gn_max,
            label_key=self.cell_type_key,
            ref_ensemble_key = ref_ensemble_key,
            query_ensemble_key =query_ensemble_key,
            inplace=inplace,
        )

        return None if inplace else result


class _BLFacade:
    """
    Thin facade over segtraq.bl bound to a SegTraQer instance.
    Methods use the parent's sdata and configured keys exclusively.
    No per-call overrides are allowed.
    """

    def __init__(self, parent: "SegTraQ") -> None:
        self._p = parent

    # ---- Global counts / summaries ----
    def num_cells(self, inplace: bool = True):
        return bl.num_cells(
            sdata=self._p.sdata,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    def num_genes(self, inplace: bool = True):
        return bl.num_genes(
            sdata=self._p.sdata,
            points_key=self._p.points_key,
            points_gene_key=self._p.points_gene_key,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    def num_transcripts(self, inplace: bool = True):
        return bl.num_transcripts(
            sdata=self._p.sdata,
            points_key=self._p.points_key,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    def perc_unassigned_transcripts(self, inplace: bool = True):
        return bl.perc_unassigned_transcripts(
            sdata=self._p.sdata,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_background_id=self._p.points_background_id,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    def genes_per_cell(self, inplace: bool = True):
        return bl.genes_per_cell(
            sdata=self._p.sdata,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_gene_key=self._p.points_gene_key,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    def transcripts_per_cell(self, inplace: bool = True):
        return bl.transcripts_per_cell(
            sdata=self._p.sdata,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    def morphological_features(self, features_to_compute: list | None = None, n_jobs: int = 1, inplace: bool = True):
        return bl.morphological_features(
            sdata=self._p.sdata,
            shapes_key=self._p.shapes_key,
            shapes_cell_id_key=self._p.shapes_cell_id_key,
            features_to_compute=features_to_compute,
            n_jobs=n_jobs,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    def transcript_density(self, inplace: bool = False):
        tavk = self._p.tables_area_volume_key
        if tavk is None:
            warnings.warn(
                "Transcript density cannot be computed because 'tables_area_volume_key' is None. "
                "Provide a cell area/volume column when initializing SegTraQ.",
                UserWarning,
                stacklevel=2,
            )
            return None
        return bl.transcript_density(
            sdata=self._p.sdata,
            tables_key=self._p.tables_key,
            points_key=self._p.points_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            tables_area_volume_key=tavk,
            inplace=inplace,
        )



    # def run_umap(self, inplace: bool = True):
    #     adata = self.sdata.tables[self.table_key]

    #     if inplace:
    #         sc.pp.pca(adata)
    #         sc.pp.neighbors(adata)
    #         sc.tl.umap(adata)
    #         return None
    #     else:
    #         adata_copy = adata.copy()
    #         sc.pp.pca(adata_copy)
    #         sc.pp.neighbors(adata_copy)
    #         sc.tl.umap(adata_copy)
    #         return adata_copy

    # def run_nuclear_correlation(
    #     self,
    #     inplace: bool = True,
    # ):
    #     tbl = self.sdata.tables[self.table_key]

    #     # 1) IoU between cell and its best-matching nucleus
    #     ious = nc.nuclear_correlation.compute_cell_nuc_ious(
    #         sdata=self.sdata,
    #         cell_id_key_shape=self.cell_key_shapes,
    #         cell_shape_key=self.shape_key_nc,
    #         nuc_shape_key=self.nuc_shape_key,
    #         n_jobs=-1,
    #         use_progress=True,
    #     ).set_index(self.cell_key_tables)

    #     if inplace:
    #         tbl.obs = tbl.obs.merge(
    #             ious.reset_index(),
    #             how="left",
    #             left_on=self.cell_key_tables,
    #             right_on=self.cell_key_tables,
    #         )

    #     print("IoU computed.")

    #     # 2) Cell–nucleus correlation per cell
    #     cell_nuc_corr = nc.nuclear_correlation.compute_cell_nuc_correlation(
    #         sdata=self.sdata,
    #         table_key=self.table_key,
    #         cell_id_key=self.cell_key_tables,
    #         transcripts_key=self.points_key,
    #         nucleus_by=self.nuc_shape_key,
    #         feature_column=self.gene_key,
    #         x_coordinate=self.x,
    #         y_coordinate=self.y,
    #         cell_shape_key=self.shape_key_nc,
    #     )
    #     cell_nuc_corr = cell_nuc_corr.set_index(self.cell_key_tables).rename(
    #         columns={
    #             "correlation": "corr_nc_cell",
    #         }
    #     )
    #     print("Cell-nuc correlation computed.")

    #     # 3) Correlation between nuclear-overlap part of cell vs rest of cell
    #     parts_corr = (
    #         nc.nuclear_correlation.compute_correlation_between_parts(
    #             sdata=self.sdata,
    #             table_key=self.table_key,
    #             cell_id_key_shape=self.cell_key_shapes,
    #             cell_shape_key=self.shape_key_nc,
    #             nuc_shape_key=self.nuc_shape_key,
    #             transcripts_key=self.points_key,
    #             feature_column=self.gene_key,
    #             x_coordinate=self.x,
    #             y_coordinate=self.y,
    #         )
    #         .set_index(self.cell_key_tables)
    #         .rename(
    #             columns={
    #                 "correlation_parts": "corr_cell_parts",
    #             }
    #         )
    #     )
    #     print("Correlation between parts correlation computed.")

    #     if inplace:
    #         to_join = (
    #             cell_nuc_corr.loc[:, ["corr_nc_cell"]]
    #             .join(parts_corr.loc[:, ["corr_cell_parts"]], how="outer")
    #             .reset_index()
    #         )
    #         tbl.obs = tbl.obs.merge(
    #             to_join,
    #             how="left",
    #             left_on=self.cell_key_tables,
    #             right_on=self.cell_key_tables,
    #         )
    #         return None
    #     else:
    #         return {
    #             "cell_nuc_correlation": cell_nuc_corr.reset_index(),
    #             "ious": ious.reset_index(),
    #             "parts_correlation": parts_corr.reset_index(),
    #         }

    # def run_clustering_stability(
    #     self,
    #     inplace: bool = True,
    #     resolution: float | tuple[float, ...] = (0.6, 0.8, 1.0),
    #     key_prefix: str = "leiden_subset",
    #     n_genes_subset: int = 100,
    #     metric: str = "euclidean",
    #     ncomps: int = 30,
    #     random_state: int = 42,
    # ):
    #     rmsd = cs.clustering_stability.compute_rmsd(
    #         self.sdata, resolution=resolution, key_prefix=key_prefix, random_state=random_state
    #     )
    #     print("RMSD computed.")
    #     sil = cs.clustering_stability.compute_silhouette_score(
    #         self.sdata,
    #         resolution=resolution,
    #         metric=metric,
    #         ncomps=ncomps,
    #         key_prefix=key_prefix,
    #         random_state=random_state,
    #     )
    #     print("Silhouette Score computed.")
    #     ari = cs.clustering_stability.compute_ari(
    #         self.sdata,
    #         resolution=(resolution if isinstance(resolution, int | float) else 1.0),
    #         n_genes_subset=n_genes_subset,
    #         key_prefix=key_prefix,
    #     )
    #     print("ARI computed.")
    #     purity = cs.clustering_stability.compute_purity(
    #         self.sdata,
    #         resolution=(resolution if isinstance(resolution, int | float) else 1.0),
    #         n_genes_subset=n_genes_subset,
    #         key_prefix=key_prefix,
    #     )
    #     print("Purity computed.")
    #     result = {"rmsd": float(rmsd), "silhouette": float(sil), "ari": float(ari), "purity": float(purity)}

    #     tbl = self.sdata.tables[self.table_key]
    #     to_drop = [
    #         c for c in tbl.obs.columns if (c.startswith("leiden_subset_100") or c.startswith("leiden_subset_allgenes"))
    #     ]
    #     tbl.obs.drop(columns=to_drop, inplace=True, errors="ignore")

    #     if inplace:
    #         tbl.uns.setdefault("segtraq", {}).setdefault("cs", {}).update(result)
    #         return None
    #     else:
    #         return result

    # def run_supervised_spillover_metrics(self, inplace: bool = True):
    #     tbl = self.sdata.tables[self.table_key]
    #     common_genes = tbl.var_names[tbl.var_names.isin(self.adata_ref.var_names)]
    #     self.adata_ref = self.adata_ref[:, common_genes].copy()

    #     markers_dict = sp.find_markers_cellspa(self.adata_ref, self.ref_celltype_key)
    #     print("Identified positive and negative markers.")

    #     # mut_markers_dict = sp.find_mutually_exclusive_genes(self.adata_ref, markers_dict, self.ref_celltype_key)
    #     # - this blows up memory - TODO find out how to store in sdata differently
    #     # print("Identified mutually exclusive markers.")

    #     # mecr = sp.compute_MECR(self.sdata, mut_markers_dict)
    #     # print("Computed mutually exclusive co-expression rate.")

    #     if "transferred_celltype" not in tbl.obs.columns:
    #         self.run_annotation()

    #     purity_results = sp.calculate_marker_purity(self.sdata, "transferred_celltype", markers_dict)
    #     print("Computed signal purity scores.")

    #     if inplace:
    #         tbl.obs = tbl.obs.merge(
    #             purity_results.reset_index(),
    #             how="left",
    #             left_on="cell_id",
    #             right_on="cell_id",
    #         )

    #         # mecr_flat = {f"{g1}__{g2}": float(v) for (g1, g2), v in mecr.items()}
    #         # tbl.uns.setdefault("segtraq", {})["sp"] = mecr_flat

    #     else:
    #         return purity_results  # , mecr

    # def run_rastering(self):
    #     pl.save_mask_to_tiff(
    #         self.sdata,
    #         labels_keys=self.label_keys,
    #         output_dir=self.out_path,
    #         palette=self.palette,
    #         unassigned_cell_id=self.background_cell_id,
    #     )

    # def run_all(self):
    #     self.run_baseline()
    #     self.run_annotation()
    #     self.run_umap()
    #     # self.run_nuclear_correlation()
    #     self.run_clustering_stability()
    #     self.run_supervised_spillover_metrics()
    #     self.run_rastering()
