import copy
import warnings
from collections.abc import Callable

import numpy as np
import spatialdata as sd
from anndata import AnnData

from . import bl, cs, ps, rc, sp, vl
from .utils import filter_cells as _filter_cells
from .utils import run_label_transfer as _run_label_transfer
from .utils import validate_spatialdata


class SegTraQ:
    def __init__(
        self,
        sdata: sd.SpatialData,
        images_key: str | None = "morphology_focus",
        tables_key: str = "table",
        tables_cell_id_key: str = "cell_id",
        tables_area_volume_key: str | None = "cell_area",
        tables_centroid_x_key: str | None = "x_centroid",
        tables_centroid_y_key: str | None = "y_centroid",
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
        nucleus_shapes_cell_id_key: str | None = None,
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

        tables_centroid_x_key : str or None, optional, default="x_centroid"
            Column in the cell table with the x-coordinate of the cell centroid.

        tables_centroid_y_key : str or None, optional, default="y_centroid"
            Column in the cell table with the y-coordinate of the cell centroid.

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

        nucleus_shapes_cell_id_key : str or None, optional, default=None
            Column linking nucleus polygons to cell IDs. If `None` but
            `nucleus_shapes_key` is provided, the shape index is used as the cell ID.

        Notes
        -----
        After initializing a SegTraQ instance, all SegTraQ modules can be run
        directly from the object using its module facades.

        For example:

        .. code-block:: python

            st = SegTraQ(sdata, ...)

            st.bl.genes_per_cell()

            st.nc.compute_cell_nuc_ious()

            st.ps.centroid_mean_coord_diff("ERBB2")

            st.sp.calculate_contamination(markers=...)

        Wrappers (run_baseline, run_nuclear_correlation, etc.) to run all metrics of a module are provided below.
        """

        # Validate spatialdata object
        validate_spatialdata(
            sdata,
            images_key=images_key,
            tables_key=tables_key,
            tables_cell_id_key=tables_cell_id_key,
            tables_area_volume_key=tables_area_volume_key,
            tables_centroid_x_key=tables_centroid_x_key,
            tables_centroid_y_key=tables_centroid_y_key,
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
        )

        self.sdata = sdata

        self.images_key = images_key

        self.tables_key = tables_key
        self.tables_cell_id_key = tables_cell_id_key
        self.tables_area_volume_key = tables_area_volume_key
        self.tables_centroid_x_key = tables_centroid_x_key
        self.tables_centroid_y_key = tables_centroid_y_key

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

        self.bl = _BLFacade(self)
        self.rc = _RCFacade(self)
        self.cs = _CSFacade(self)
        self.vl = _VLFacade(self)
        self.sp = _SPFacade(self)
        self.ps = _PSFacade(self)

    @property
    def sdata(self):
        """Underlying SpatialData object (modifiable)."""
        return self._sdata

    @sdata.setter
    def sdata(self, value):
        if not isinstance(value, sd.SpatialData):
            raise TypeError("Must be a SpatialData object")
        self._sdata = value

    def run_baseline(self, inplace: bool = True):
        """
        Compute baseline SegTraQ metrics (via the bound BL facade) and optionally
        merge them into the cell table.

        Metrics
        -------
        - genes_per_cell
        - transcripts_per_cell
        - mean_transcripts_per_gene_per_cell
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
            `summary`, `genes_per_cell`, `transcripts_per_cell`,
                `mean_transcripts_per_gene_per_cell`and optionally `transcript_density`.
        """

        gpc = self.bl.genes_per_cell(inplace=inplace)
        tpc = self.bl.transcripts_per_cell(inplace=inplace)
        tgc = self.bl.mean_transcripts_per_gene_per_cell(inplace=inplace)
        mrp = self.bl.morphological_features(inplace=inplace)

        dens_raw = self.bl.transcript_density(inplace=inplace)
        dens = None if dens_raw is None else dens_raw

        summary = dict(
            num_cells=self.bl.num_cells(inplace=inplace),
            num_genes=self.bl.num_genes(inplace=inplace),
            num_transcripts=self.bl.num_transcripts(inplace=inplace),
            perc_unassigned_transcripts=self.bl.perc_unassigned_transcripts(inplace=inplace),
            perc_unassigned_transcripts_per_gene=self.bl.perc_unassigned_transcripts_per_gene(inplace=inplace),
        )

        if inplace:
            return None
        else:
            out = {
                "summary": summary,
                "genes_per_cell": gpc,
                "transcripts_per_cell": tpc,
                "mean_transcripts_per_gene_per_cell": tgc,
                "morphological_feautres": mrp,
            }
            if dens is not None:
                out["transcript_density"] = dens
            return out

    def run_region_correlation(self, metric: str = "cosine_sim", n_jobs: int = -1, inplace: bool = True):
        """
        Compute region-correlation metrics and optionally merge them into the cell table.

        This runs, in order:
        1) IoU between each cell and its best-matching nucleus
        2) Correlation between per-cell expression and its matched nucleus (Pearson)
        3) Correlation between the cell's nucleus-overlap part vs. remainder (vectorized)
        4) Compute correlation of gene expression in an eroded interior ("center") and
           a thin outer shell ("border"), and (2) comparing the border with the neighborhood
           composition vector (NCV).

        Parameters
        ----------
        inplace : bool, default=True
            If True, writes results into `sdata.tables[tables_key].obs` and returns None.
            If False, returns a dictionary of DataFrames without writing.

        Returns
        -------
        None or dict
        - If `inplace=True`: returns None after writing to `sdata`.
        - If `inplace=False`: returns a dict with keys:
        * "ious"                  : DataFrame with columns [tables_cell_id_key, best_nuc_id, IoU]
        * "cell_nuc_correlation". : DataFrame with columns [tables_cell_id_key, best_nuc_id, IoU, corr_nc_cell]
        * "parts_correlation"     : DataFrame with columns [tables_cell_id_key, best_nuc_id, IoU, corr_cell_parts]
        * "center_border_ncv_corr": DataFrame with columns
                                    [tables_cell_id_key, corr_center_border, corr_border_ncv, corr_ncv_vs_center]

        Notes
        -----
        - Requires `self.nucleus_shapes_key` (nucleus boundaries).
        """
        assert self.nucleus_shapes_key is not None, (
            "Cannot run region correlation: `nucleus_shapes_key` is None. "
            "Define the nucleus shape layer when initializing SegTraQ."
        )

        ious = self.rc.compute_cell_nuc_ious(n_jobs=n_jobs, inplace=inplace)
        cell_nuc_corr = self.rc.compute_cell_nuc_correlation(metric=metric, n_jobs_iou=n_jobs, inplace=inplace)
        parts_corr = self.rc.compute_correlation_between_parts(metric=metric, n_jobs=n_jobs, inplace=inplace)
        center_border_ncv_corr = self.rc.compute_center_border_ncv_correlation(metric=metric, inplace=inplace)

        if inplace:
            return None

        else:
            return {
                "ious": ious,
                "cell_nuc_correlation": cell_nuc_corr,
                "parts_correlation": parts_corr,
                "center_border_ncv_corr": center_border_ncv_corr,
            }

    def run_label_transfer(
        self,
        adata_ref=AnnData,
        tx_min: float = 10.0,
        tx_max: float = 2000.0,
        gn_min: float = 5.0,
        gn_max: float = np.inf,
        cell_type_key: str = "transferred_cell_type",
        ref_cell_type: str = "cell_type",
        ref_ensemble_key: str | None = None,
        query_ensemble_key: str | None = "gene_ids",
        inplace: bool = True,
    ):
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
        cell_type_key : str
            Column name to store transferred labels in `.obs` when `inplace=True`.
        ref_cell_type: str, default="cell_type"
            Column name of cell-type annotations in `adata_ref.obs[ref_cell_type]`.
        ref_ensemble_key: str or None, default=None
            Column name in `adata_ref.var` that contains unique gene/ensemble IDs.
            If None, `adata_ref.var_names` will be used.
        query_ensemble_key: str or None, default="gene_ids"
            Column name in `self.sdata.tables[self.tables_key].var` that contains unique gene/ensemble IDs.
            If None, `self.sdata.tables[self.tables_key].var_names` will be used.
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
            ref_cell_type=ref_cell_type,
            tables_key=self.tables_key,
            tables_cell_id_key=self.tables_cell_id_key,
            points_key=self.points_key,
            points_cell_id_key=self.points_cell_id_key,
            points_gene_key=self.points_gene_key,
            tx_min=tx_min,
            tx_max=tx_max,
            gn_min=gn_min,
            gn_max=gn_max,
            cell_type_key=cell_type_key,
            ref_ensemble_key=ref_ensemble_key,
            query_ensemble_key=query_ensemble_key,
            inplace=inplace,
        )

        return None if inplace else result

    run_label_transfer.__doc__ = _run_label_transfer.__doc__

    def run_volume(self, inplace: bool = True):
        """
        Run volume metrics for SegTraQ.
        """
        sim_top_bottom = self.vl.compute_sim_top_bottom_z(inplace=inplace)
        heterotypic_overlap = self.vl.compute_heterotypic_overlap_fraction(inplace=inplace)
        mean_vsi = self.vl.compute_mean_vsi_per_cell(inplace=inplace)
        if inplace:
            return None
        else:
            return {
                "cosine_sim_top_bottom_z": sim_top_bottom,
                "heterotypic_overlap": heterotypic_overlap,
                "mean_vsi": mean_vsi,
            }

    def run_supervised_metrics(
        self,
        markers: dict[str, dict[str, list[str]]],
        mutually_exclusive_pairs: list[tuple[str, str]],
        cell_type_key: str = "transferred_cell_type",
        use_quantiles: bool = False,
        inplace: bool = True,
    ):
        """
        Run supervised metrics (SP) for SegTraQ.

        This function executes supervised metrics that require externally
        provided markers and mutually exclusive gene pairs.

        This runs, in order:
            1) MECR per mutually-exclusive gene pair
            2) Spatial contamination (directional leakage)
            3) Per-cell marker purity
            4) Differential abundance between bordering/non-bordering cells

        Parameters
        ----------
        markers : dict
            Required. Precomputed marker dictionary:
            {cell_type: {"positive": [...], "negative": [...]}}

        mutually_exclusive_pairs : list of tuple
            Precomputed mutually exclusive gene pairs.

        cell_type_key : str
            Cell-type column in `sdata.tables[tables_key].obs`.

        use_quantiles : bool, optional
            Passed to SP.calculate_marker_purity.

        inplace : bool, default=True
            If True: write results into the `sdata.tables[...]` object.
            If False: return a dictionary of all computed results.

        Returns
        -------
        None or dict
            - None if inplace=True
            - dict with keys:
                {
                "MECR": ...,
                "contamination": ...,
                "marker_purity": ...,
                }
        """

        out = {}

        assert mutually_exclusive_pairs is not None and len(mutually_exclusive_pairs) > 0, (
            "MECR requires `mutually_exclusive_pairs`. Please compute them externally "
            "with `segtraq.get_mut_excl_markers` and pass them here."
        )
        fisher_or, fisher_pval = self.sp.compute_MECR(
            markers=markers,
            inplace=inplace,
        )

        purity_df = self.sp.calculate_marker_purity(
            cell_type_key=cell_type_key,
            markers=markers,
            use_quantiles=use_quantiles,
            inplace=inplace,
        )

        per_cell_contam, cont_mat, cont_mat_bin = self.sp.calculate_neighbor_contamination(
            cell_type_key=cell_type_key,
            markers=markers,
            inplace=inplace,
        )

        # TODO: need to add the neighbor prediction here! Should we do this for all pairs of cell types?

        if inplace:
            return None
        else:
            out = {
                "MECR": (fisher_or, fisher_pval),
                "marker_purity": purity_df,
                "neighbor_contamination": (per_cell_contam, cont_mat, cont_mat_bin),
            }
            return out

    def run_point_statistics(
        self,
        genes: str | list[str] = None,
        erosion_fraction_of_radius: float = 0.3,
        inplace: bool = True,
    ):
        """
        Compute point-statistics metrics per feature and optionally merge into the cell table.

        This runs:
        1) `centroid_mean_coord_diff`  → per-cell distance between mean transcript coords and cell centroid
        2) `distance_to_membrane`      → per-cell mean distance of transcripts to the cell boundary

        Parameters
        ----------
        feature : str
            Feature (gene) name to evaluate.
        inplace : bool, default=True
            If True, merges a compact set of columns into `sdata.tables[tables_key].obs`:
                - ps_cmd_dist__{feature}    : distance from centroid_mean_coord_diff
                - ps_dtm_dist__{feature}    : distance_to_outline from distance_to_membrane
                - ps_dtm_inv__{feature}     : distance_to_outline_inverse from distance_to_membrane
            If False, returns a dictionary with raw DataFrames for each metric and feature.

        Returns
        -------
        None or dict
            - If `inplace=True`: returns None (results written to `.obs`).
            - If `inplace=False`: returns
                {
                "centroid_mean_coord_diff": {genes: DataFrame, ...},
                "distance_to_membrane":    {genes: DataFrame, ...},
                }
        """

        cmd_df = self.ps.centroid_mean_coord_diff(genes=genes, inplace=inplace)
        dtm_df = self.ps.distance_to_membrane(genes=genes, inplace=inplace)
        pe_df = self.ps.periphery_enrichment_score(
            genes=genes, erosion_fraction_of_radius=erosion_fraction_of_radius, inplace=inplace
        )

        if inplace:
            return None
        else:
            if genes is None:
                feature = "all_genes"
            elif isinstance(genes, str):
                feature = genes
            else:
                feature = f"{len(genes)}_genes"
            out = {
                f"centroid_mean_coord_diff_{feature}": cmd_df[f"distance_{feature}"],
                f"distance_to_membrane_{feature}": dtm_df[f"distance_to_outline_{feature}"],
                f"distance_to_outline_inverse_{feature}": dtm_df[f"distance_to_outline_inverse_{feature}"],
                f"periphery_enrichment_score_{feature}": pe_df[f"periphery_enrichment_score_{feature}"],
            }
            return out

    def filter_cells(
        self,
        col: str,
        func: Callable,
        inplace: bool = True,
    ):
        """
        Filter cells from the cell table based on a user-defined function.

        Parameters
        ----------
        col : str
            Column in the cell table to apply the filtering function on.
        func : Callable
            A function that takes a single argument (the column value) and returns
            True if the cell should be kept, False otherwise.
        inplace : bool, default=True
            If True, modifies `self.sdata` in place.
            If False, returns a new SpatialData object with the filtered cells.

        Returns
        -------
        None or SpatialData
            - If `inplace=True`: returns None after modifying `self.sdata`.
            - If `inplace=False`: returns a new SpatialData object with filtered cells.

        Example
        -------
        >>> st.filter_cells(col='cell_area', func=lambda x: x > 100)
        """
        adata = _filter_cells(
            adata=self.sdata.tables[self.tables_key],
            col=col,
            func=func,
        )

        # synchronizing the rest of the sdata object to the now filtered table
        # (removes e.g. shapes whose cell_id is gone after filtering)
        if not inplace:
            sdata = copy.deepcopy(self.sdata)
        else:
            sdata = self.sdata

        assert adata.n_obs > 0, "Filtering removed all cells; no cells remain after filtering."

        sdata.tables[self.tables_key] = adata
        sdata = sd.match_sdata_to_table(sdata, "table")

        if inplace:
            self.sdata = sdata
            return None
        return sdata

    filter_cells.__doc__ = filter_cells.__doc__


class _BLFacade:
    """
    Thin facade over segtraq.bl bound to a SegTraQ instance.
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

    num_cells.__doc__ = bl.num_cells.__doc__

    def num_genes(self, inplace: bool = True):
        return bl.num_genes(
            sdata=self._p.sdata,
            points_key=self._p.points_key,
            points_gene_key=self._p.points_gene_key,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    num_genes.__doc__ = bl.num_genes.__doc__

    def num_transcripts(self, inplace: bool = True):
        return bl.num_transcripts(
            sdata=self._p.sdata,
            points_key=self._p.points_key,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    num_transcripts.__doc__ = bl.num_transcripts.__doc__

    def perc_unassigned_transcripts(self, inplace: bool = True):
        return bl.perc_unassigned_transcripts(
            sdata=self._p.sdata,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_background_id=self._p.points_background_id,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    perc_unassigned_transcripts.__doc__ = bl.perc_unassigned_transcripts.__doc__

    def genes_per_cell(self, inplace: bool = True):
        return bl.genes_per_cell(
            sdata=self._p.sdata,
            tables_cell_id_key=self._p.tables_cell_id_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_gene_key=self._p.points_gene_key,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    genes_per_cell.__doc__ = bl.genes_per_cell.__doc__

    def mean_transcripts_per_gene_per_cell(self, inplace: bool = True):
        return bl.mean_transcripts_per_gene_per_cell(
            sdata=self._p.sdata,
            tables_cell_id_key=self._p.tables_cell_id_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_gene_key=self._p.points_gene_key,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    mean_transcripts_per_gene_per_cell.__doc__ = bl.mean_transcripts_per_gene_per_cell.__doc__

    def perc_unassigned_transcripts_per_gene(self, inplace: bool = True):
        return bl.perc_unassigned_transcripts_per_gene(
            sdata=self._p.sdata,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_background_id=self._p.points_background_id,
            points_gene_key=self._p.points_gene_key,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    perc_unassigned_transcripts_per_gene.__doc__ = bl.perc_unassigned_transcripts_per_gene.__doc__

    def transcripts_per_cell(self, inplace: bool = True):
        return bl.transcripts_per_cell(
            sdata=self._p.sdata,
            tables_cell_id_key=self._p.tables_cell_id_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    transcripts_per_cell.__doc__ = bl.transcripts_per_cell.__doc__

    def morphological_features(self, features_to_compute: list | None = None, n_jobs: int = 1, inplace: bool = True):
        return bl.morphological_features(
            sdata=self._p.sdata,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_key=self._p.shapes_key,
            shapes_cell_id_key=self._p.shapes_cell_id_key,
            features_to_compute=features_to_compute,
            n_jobs=n_jobs,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    morphological_features.__doc__ = bl.morphological_features.__doc__

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

    transcript_density.__doc__ = bl.transcript_density.__doc__


class _RCFacade:
    """
    Bound region-correlation (rc) metrics interface for a SegTraQ instance.
    Methods use the parent's `sdata` and configured keys.
    No per-call overrides are allowed.
    """

    def __init__(self, parent: "SegTraQ") -> None:
        self._p = parent

    def compute_cell_nuc_ious(self, n_jobs: int = -1, inplace: bool = True):
        assert self._p.nucleus_shapes_key is not None, (
            "Cannot compute IoUs: `nucleus_shapes_key` is None. "
            "Define a valid nucleus shape layer in `SegTraQ` before running `nc` metrics."
        )
        return rc.compute_cell_nuc_ious(
            sdata=self._p.sdata,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_key=self._p.shapes_key,
            shapes_cell_id_key=self._p.shapes_cell_id_key,
            nucleus_shapes_key=self._p.nucleus_shapes_key,
            nucleus_shapes_cell_id_key=self._p.nucleus_shapes_cell_id_key,
            n_jobs=n_jobs,
            use_progress=True,
            inplace=inplace,
        )

    compute_cell_nuc_ious.__doc__ = rc.compute_cell_nuc_ious.__doc__

    def compute_cell_nuc_correlation(self, metric: str = "cosine_sim", n_jobs_iou: int = -1, inplace: bool = True):
        assert self._p.nucleus_shapes_key is not None, (
            "Cannot compute IoUs: `nucleus_shapes_key` is None. "
            "Define a valid nucleus shape layer in `SegTraQ` before running `nc` metrics."
        )
        return rc.compute_cell_nuc_correlation(
            sdata=self._p.sdata,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_key=self._p.shapes_key,
            shapes_cell_id_key=self._p.shapes_cell_id_key,
            nucleus_shapes_key=self._p.nucleus_shapes_key,
            nucleus_shapes_cell_id_key=self._p.nucleus_shapes_cell_id_key,
            points_key=self._p.points_key,
            points_gene_key=self._p.points_gene_key,
            metric=metric,
            n_jobs_iou=n_jobs_iou,
            inplace=inplace,
        )

    compute_cell_nuc_correlation.__doc__ = rc.compute_cell_nuc_correlation.__doc__

    def compute_correlation_between_parts(self, metric: str = "cosine_sim", n_jobs: int = -1, inplace: bool = True):
        assert self._p.nucleus_shapes_key is not None, (
            "Cannot compute IoUs: `nucleus_shapes_key` is None. "
            "Define a valid nucleus shape layer in `SegTraQ` before running `nc` metrics."
        )
        return rc.compute_correlation_between_parts(
            sdata=self._p.sdata,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_key=self._p.shapes_key,
            shapes_cell_id_key=self._p.shapes_cell_id_key,
            nucleus_shapes_key=self._p.nucleus_shapes_key,
            nucleus_shapes_cell_id_key=self._p.nucleus_shapes_cell_id_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_background_id=self._p.points_background_id,
            points_gene_key=self._p.points_gene_key,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            metric=metric,
            n_jobs=n_jobs,
            inplace=inplace,
        )

    compute_correlation_between_parts.__doc__ = rc.compute_correlation_between_parts.__doc__

    def compute_center_border_ncv_correlation(
        self,
        erosion_fraction_of_radius: float = 0.2,
        radius_factor: float = 2.0,
        metric: str = "cosine_sim",
        inplace: bool = True,
    ):
        return rc.compute_center_border_ncv_correlation(
            sdata=self._p.sdata,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_key=self._p.shapes_key,
            shapes_cell_id_key=self._p.shapes_cell_id_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            points_gene_key=self._p.points_gene_key,
            erosion_fraction_of_radius=erosion_fraction_of_radius,
            radius_factor=radius_factor,
            metric=metric,
            inplace=inplace,
        )

    compute_center_border_ncv_correlation.__doc__ = rc.compute_center_border_ncv_correlation.__doc__


class _SPFacade:
    """
    Bound supervised (sp) interface for a SegTraQ instance.
    Methods use the parent's `sdata` and configured `tables_key`.
    No per-call overrides are allowed.
    """

    def __init__(self, parent: "SegTraQ") -> None:
        self._p = parent

    def compute_MECR(self, markers: dict[str, dict[str, list[str]]], pseudocount: float = 0.5, inplace: bool = True):
        return sp.compute_MECR(
            sdata=self._p.sdata,
            markers=markers,
            pseudocount=pseudocount,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    compute_MECR.__doc__ = sp.compute_MECR.__doc__

    def calculate_marker_purity(
        self,
        cell_type_key: str,
        markers: dict[str, dict[str, list[str]]],
        use_quantiles: bool = False,
        require_neighbor_expression: bool = True,
        weight_cont: float = 0.7,
        neighbors_key: str | None = "spatial_connectivities",
        inplace: bool = True,
    ):
        return sp.calculate_marker_purity(
            sdata=self._p.sdata,
            cell_type_key=cell_type_key,
            markers=markers,
            use_quantiles=use_quantiles,
            require_neighbor_expression=require_neighbor_expression,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            tables_centroid_x_key=self._p.tables_centroid_x_key,
            tables_centroid_y_key=self._p.tables_centroid_y_key,
            weight_cont=weight_cont,
            neighbors_key=neighbors_key,
            inplace=inplace,
        )

    calculate_marker_purity.__doc__ = sp.calculate_marker_purity.__doc__

    def calculate_neighbor_contamination(
        self,
        cell_type_key: str,
        markers: dict[str, dict[str, list[str]]],
        require_neighbor_expression: bool = True,
        neighbors_key: str | None = "spatial_connectivities",
        uns_key: str = "negative_marker_contamination",
        uns_key_binary: str = "negative_marker_contamination_binary",
        inplace: bool = True,
    ):
        return sp.calculate_neighbor_contamination(
            sdata=self._p.sdata,
            cell_type_key=cell_type_key,
            markers=markers,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            tables_centroid_x_key=self._p.tables_centroid_x_key,
            tables_centroid_y_key=self._p.tables_centroid_y_key,
            require_neighbor_expression=require_neighbor_expression,
            neighbors_key=neighbors_key,
            uns_key=uns_key,
            uns_key_binary=uns_key_binary,
            inplace=inplace,
        )

    calculate_neighbor_contamination.__doc__ = sp.calculate_neighbor_contamination.__doc__

    def neighbor_prediction(
        self,
        ct1: str,
        ct2: str,
        cell_type_key: str = "transferred_cell_type",
        grid_shape: tuple[int, int] = (10, 10),
        n_permutations: int = 100,
        seed: int = 0,
        inplace: bool = True,
    ):
        return sp.neighbor_prediction(
            self._p.sdata,
            ct1,
            ct2,
            cell_type_key=cell_type_key,
            tables_x_key=self._p.tables_x_key,
            tables_y_key=self._p.tables_y_key,
            grid_shape=grid_shape,
            n_permutations=n_permutations,
            seed=seed,
            inplace=inplace,
        )

    neighbor_prediction.__doc__ = sp.neighbor_prediction.__doc__


class _PSFacade:
    """
    Bound points-statistics (ps) interface for a SegTraQ instance.
    Methods use the parent's `sdata` and configured keys.
    No per-call overrides are allowed.
    """

    def __init__(self, parent: "SegTraQ") -> None:
        self._p = parent

    def centroid_mean_coord_diff(self, genes: str | list[str] = None, inplace: bool = True):
        return ps.centroid_mean_coord_diff(
            sdata=self._p.sdata,
            genes=genes,
            tables_key=self._p.tables_key,
            points_gene_key=self._p.points_gene_key,
            points_key=self._p.points_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_cell_id_key=self._p.shapes_cell_id_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            shapes_key=self._p.shapes_key,
            inplace=inplace,
        )

    centroid_mean_coord_diff.__doc__ = ps.centroid_mean_coord_diff.__doc__

    def distance_to_membrane(
        self,
        genes: str | list[str] | None = None,
        cell_type_key: str | None = None,
        cell_type_query: str | None = None,
        inplace: bool = True,
    ):
        return ps.distance_to_membrane(
            sdata=self._p.sdata,
            genes=genes,
            cell_type_key=cell_type_key,
            cell_type_query=cell_type_query,
            tables_key=self._p.tables_key,
            points_gene_key=self._p.points_gene_key,
            points_key=self._p.points_key,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_cell_id_key=self._p.shapes_cell_id_key,
            points_cell_id_key=self._p.points_cell_id_key,
            inplace=inplace,
        )

    distance_to_membrane.__doc__ = ps.distance_to_membrane.__doc__

    def periphery_enrichment_score(
        self,
        genes: str | list[str] | None = None,
        erosion_fraction_of_radius: float = 0.3,
        inplace: bool = True,
    ):
        return ps.periphery_enrichment_score(
            sdata=self._p.sdata,
            genes=genes,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_key=self._p.shapes_key,
            shapes_cell_id_key=self._p.shapes_cell_id_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            points_gene_key=self._p.points_gene_key,
            erosion_fraction_of_radius=erosion_fraction_of_radius,
            inplace=inplace,
        )

    periphery_enrichment_score.__doc__ = ps.periphery_enrichment_score.__doc__


class _CSFacade:
    """
    Thin facade over segtraq.cs bound to a SegTraQ instance.
    Methods use the parent's sdata and configured keys exclusively.
    No per-call overrides are allowed.
    """

    def __init__(self, parent: "SegTraQ") -> None:
        self._p = parent

    def compute_rmsd(
        self,
        resolution: float | list[float] = (0.6, 0.8, 1.0),
        key_prefix: str = "leiden_subset",
        random_state: int = 42,
        cell_type_key: str | None = None,
        inplace: bool = True,
    ) -> float:
        return cs.compute_rmsd(
            self._p.sdata,
            resolution=resolution,
            key_prefix=key_prefix,
            random_state=random_state,
            cell_type_key=cell_type_key,
            inplace=inplace,
        )

    def compute_mean_cosine_distance(
        self,
        resolution: float | list[float] = (0.6, 0.8, 1.0),
        key_prefix: str = "leiden_subset",
        random_state: int = 42,
        cell_type_key: str | None = None,
        inplace: bool = True,
    ) -> float:
        return cs.compute_mean_cosine_distance(
            self._p.sdata,
            resolution=resolution,
            key_prefix=key_prefix,
            random_state=random_state,
            cell_type_key=cell_type_key,
            inplace=inplace,
        )

    def compute_silhouette_score(
        self,
        resolution: float | list[float] = (0.6, 0.8, 1.0),
        metric: str = "euclidean",
        key_prefix: str = "leiden_subset",
        random_state: int = 42,
        cell_type_key: str | None = None,
        inplace: bool = True,
    ) -> float:
        return cs.compute_silhouette_score(
            self._p.sdata,
            resolution=resolution,
            metric=metric,
            key_prefix=key_prefix,
            random_state=random_state,
            cell_type_key=cell_type_key,
            inplace=inplace,
        )

    def compute_purity(
        self,
        resolution: float = 1.0,
        frac_cells_subset: float = 0.63,
        key_prefix: str = "leiden_subset",
        inplace: bool = True,
    ) -> float:
        return cs.compute_purity(
            self._p.sdata,
            resolution=resolution,
            frac_cells_subset=frac_cells_subset,
            key_prefix=key_prefix,
            inplace=inplace,
        )

    def compute_ari(
        self,
        resolution: float = 1.0,
        frac_cells_subset: float = 0.63,
        key_prefix: str = "leiden_subset",
        inplace: bool = True,
    ) -> float:
        return cs.compute_ari(
            self._p.sdata,
            resolution=resolution,
            frac_cells_subset=frac_cells_subset,
            key_prefix=key_prefix,
            inplace=inplace,
        )


class _VLFacade:
    """
    Thin facade over segtraq.vl bound to a SegTraQ instance.
    Methods use the parent's sdata and configured keys exclusively.
    No per-call overrides are allowed.
    """

    def __init__(self, parent: "SegTraQ") -> None:
        self._p = parent

    def compute_sim_top_bottom_z(
        self,
        correct_z_drift: bool = True,
        max_points: int = 1_000_000,
        q0: float = 0.01,
        q1: float = 0.99,
        seed: int | None = 0,
        q: float = 0.30,
        scale: float = 1e4,
        min_genes: int = 5,
        min_transcripts: int = 10,
        inplace: bool = True,
    ):
        return vl.compute_sim_top_bottom_z(
            self._p.sdata,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            points_key=self._p.points_key,
            points_background_id=self._p.points_background_id,
            points_cell_id_key=self._p.points_cell_id_key,
            points_gene_key=self._p.points_gene_key,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            points_z_key=self._p.points_z_key,
            correct_z_drift=correct_z_drift,
            q0=q0,
            q1=q1,
            seed=seed,
            q=q,
            scale=scale,
            min_genes=min_genes,
            min_transcripts=min_transcripts,
            inplace=inplace,
        )

    def compute_heterotypic_overlap_fraction(
        self,
        cell_type_key: str = "transferred_cell_type",
        shapes_key_list: list[str] = (
            "cell_boundaries_z0",
            "cell_boundaries_z1",
            "cell_boundaries_z2",
            "cell_boundaries_z3",
        ),
        unknown_label: str = "Unknown",
        unknown_policy: str = "treat_as_label",
        inplace: bool = True,
    ):
        return vl.compute_heterotypic_overlap_fraction(
            sdata=self._p.sdata,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_cell_id_key=self._p.shapes_cell_id_key,
            cell_type_key=cell_type_key,
            shapes_key_list=shapes_key_list,
            unknown_label=unknown_label,
            unknown_policy=unknown_policy,
            inplace=inplace,
        )

    def compute_mean_vsi_per_cell(
        self,
        vsi_map: np.ndarray,
        shift_to_origin: bool = True,
        inplace: bool = True,
    ):
        return vl.compute_mean_vsi_per_cell(
            sdata=self._p.sdata,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            points_key=self._p.points_key,
            points_background_id=self._p.points_background_id,
            points_cell_id_key=self._p.points_cell_id_key,
            points_gene_key=self._p.points_gene_key,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            vsi_map=vsi_map,
            shift_to_origin=shift_to_origin,
            inplace=inplace,
        )
