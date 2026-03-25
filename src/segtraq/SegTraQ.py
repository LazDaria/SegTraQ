from collections.abc import Callable
from typing import Literal

import numpy as np
import spatialdata as sd
from anndata import AnnData

from . import bl, cs, ps, rs, sp, vl
from .utils import _filter_control_and_low_quality_transcripts, validate_spatialdata
from .utils import filter_cells as _filter_cells
from .utils import run_label_transfer as _run_label_transfer


class SegTraQ:
    def __init__(
        self,
        sdata: sd.SpatialData,
        images_key: str | None = "morphology_focus",
        tables_key: str = "table",
        tables_cell_id_key: str = "cell_id",
        tables_area_key: str | None = "cell_area",
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
        shapes_cell_id_key: str = "cell_id",
        nucleus_shapes_key: str | None = "nucleus_boundaries",
        nucleus_shapes_cell_id_key: str = "cell_id",
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

        tables_area_key : str or None, optional, default="cell_area"
            Column in the cell table with cell area (2D).
            If `None`, area will be computed via
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

        shapes_cell_id_key : str, optional, default="cell_id"
            Cell ID key for `sdata.shapes[shapes_key]`. Must match either the shapes index name
            or a column name (which will be set as the index if needed).

        nucleus_shapes_key : str or None, optional, default="nucleus_boundaries"
            Key in `sdata.shapes` for nucleus boundary polygons, if available.
            If None, a nucleus mask can be obtained via `segtraq.run_cellpose`.

        nucleus_shapes_cell_id_key : str, optional, default="cell_id"
            Cell ID key for `sdata.shapes[nucleus_shapes_key]`. Must match either the shapes
            index name or a column name (which will be set as the index if needed).

        Notes
        -----
        After initializing a SegTraQ instance, all SegTraQ modules can be run
        directly from the object using its module facades.

        Wrappers (run_baseline, run_nuclear_correlation, etc.) to run all metrics of a module are provided below.
        """

        # Validate spatialdata object
        validate_spatialdata(
            sdata,
            images_key=images_key,
            tables_key=tables_key,
            tables_cell_id_key=tables_cell_id_key,
            tables_area_key=tables_area_key,
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

        # if these are set to None, the validate_spatialdata automatically computes them
        self.tables_area_key = tables_area_key if tables_area_key is not None else "cell_area"
        self.tables_centroid_x_key = tables_centroid_x_key if tables_centroid_x_key is not None else "centroid_x"
        self.tables_centroid_y_key = tables_centroid_y_key if tables_centroid_y_key is not None else "centroid_y"

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

        self.bl = _BLFacade(self)
        self.rs = _RSFacade(self)
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

    def run_baseline(
        self,
        inplace: bool = True,
        *,
        morphological_kwargs: dict | None = None,
        image_kwargs: dict | None = None,
    ):
        """
        Run baseline (bl) metrics.

        Convenience wrapper around global and per-cell summary metrics. Runs, in order:

        1) number of cells
        2) number of transcripts
        3) number of genes
        4) % unassigned transcripts
        5) % unassigned transcripts per gene
        6) transcripts per cell
        7) genes per cell
        8) mean transcripts per detected gene per cell
        9) morphological features
        10) transcript density
        11) image features (if an image is present in the dataset)

        Parameters
        ----------
        inplace : bool, default=True
            If True, results are merged into `.uns`, `.obs`, and/or `.var` as implemented
            by each metric, and None is returned.
            If False, per-metric results are returned in a dict.
        morphological_kwargs : dict or None, optional
            Extra arguments forwarded to :meth:`bl.morphological_features`.
        image_kwargs : dict or None, optional
            Extra arguments forwarded to :meth:`bl.image_features`.

        Returns
        -------
        None or dict
            If ``inplace=True``, returns None.
            If ``inplace=False``, returns a dict with keys:
            - ``"num_cells"``
            - ``"num_transcripts"``
            - ``"num_genes"``
            - ``"perc_unassigned_transcripts"``
            - ``"perc_unassigned_transcripts_per_gene"``
            - ``"transcripts_per_cell"``
            - ``"genes_per_cell"``
            - ``"mean_transcripts_per_gene_per_cell"``
            - ``"morphological_features"``
            - ``"transcript_density"``
            - ``"image_features"``
        """
        morphological_kwargs = {} if morphological_kwargs is None else dict(morphological_kwargs)

        nc = self.bl.num_cells(inplace=inplace)
        nt = self.bl.num_transcripts(inplace=inplace)
        ng = self.bl.num_genes(inplace=inplace)
        pu = self.bl.perc_unassigned_transcripts(inplace=inplace)

        pu_pg = self.bl.perc_unassigned_transcripts_per_gene(inplace=inplace)

        tpc = self.bl.transcripts_per_cell(inplace=inplace)
        gpc = self.bl.genes_per_cell(inplace=inplace)
        mtg = self.bl.mean_transcripts_per_gene_per_cell(inplace=inplace)
        dens = self.bl.transcript_density(inplace=inplace)

        morph = self.bl.morphological_features(inplace=inplace, **(morphological_kwargs))
        
        res = {
            "num_cells": nc,
            "num_transcripts": nt,
            "num_genes": ng,
            "perc_unassigned_transcripts": pu,
            "perc_unassigned_transcripts_per_gene": pu_pg,
            "transcripts_per_cell": tpc,
            "genes_per_cell": gpc,
            "mean_transcripts_per_gene_per_cell": mtg,
            "morphological_features": morph,
            "transcript_density": dens,
        }
        
        if self.images_key is not None:
            image_kwargs = {} if image_kwargs is None else dict(image_kwargs)
            img_feats = self.bl.image_features(inplace=inplace, **image_kwargs)
            res["image_features"] = img_feats

        if inplace:
            return None

        return res

    def run_region_similarity(
        self,
        metric: str = "cosine_sim",
        n_jobs: int = -1,
        inplace: bool = True,
        iou_kwargs: dict = None,
        similarity_nucleus_cell_kwargs: dict = None,
        similarity_nucleus_cytoplasm_kwargs: dict = None,
        similarity_border_neighborhood_kwargs: dict = None,
    ):
        """
        Compute region similarity metrics and optionally merge them into the cell table.

        This runs, in order:
        1) IoU between each cell and its best-matching nucleus
        2) Similarity between per-cell expression and its matched nucleus
        3) Similarity between the cell's nucleus vs. cytoplasm expression
        4) Similarity of gene expression in an eroded interior ("center") and
           a thin outer shell ("border"), and (2) comparing the border with the neighborhood.

        Parameters
        ----------
        metric : str, default="cosine_sim"
        n_jobs : int, default=-1
        inplace : bool, default=True
            If True, writes results into `sdata.tables[tables_key].obs` and returns None.
            If False, returns a dictionary of DataFrames without writing.
        iou_kwargs : dict, optional
            Additional keyword arguments to pass to `match_nuclei_to_cells`.
        similarity_nucleus_cell_kwargs : dict, optional
            Additional keyword arguments to pass to `similarity_nucleus_cell`.
        similarity_nucleus_cytoplasm_kwargs : dict, optional
            Additional keyword arguments to pass to `similarity_nucleus_cytoplasm`.
        similarity_border_neighborhood_kwargs : dict, optional
            Additional keyword arguments to pass to `similarity_border_neighborhood`.

        Returns
        -------
        None or dict
        - If `inplace=True`: returns None after writing to `sdata`.
        - If `inplace=False`: returns a dict with keys:
        * "ious": pd.DataFrame
        * "similarity_nucleus_cell": pd.DataFrame
        * "similarity_nucleus_cytoplasm": pd.DataFrame
        * "similarity_border_neighborhood": pd.DataFrame

        Notes
        -----
        - Requires `self.nucleus_shapes_key` (nucleus boundaries).
        """
        ious = self.rs.match_nuclei_to_cells(n_jobs=n_jobs, inplace=inplace, **(iou_kwargs or {}))
        similarity_nucleus_cell = self.rs.similarity_nucleus_cell(
            metric=metric, n_jobs=n_jobs, inplace=inplace, **(similarity_nucleus_cell_kwargs or {})
        )
        similarity_nucleus_cytoplasm = self.rs.similarity_nucleus_cytoplasm(
            metric=metric, n_jobs=n_jobs, inplace=inplace, **(similarity_nucleus_cytoplasm_kwargs or {})
        )
        similarity_border_neighborhood = self.rs.similarity_border_neighborhood(
            metric=metric, inplace=inplace, **(similarity_border_neighborhood_kwargs or {})
        )

        if inplace:
            return None

        else:
            return {
                "ious": ious,
                "similarity_nucleus_cell": similarity_nucleus_cell,
                "similarity_nucleus_cytoplasm": similarity_nucleus_cytoplasm,
                "similarity_border_neighborhood": similarity_border_neighborhood,
            }

    def run_volume_metrics(
        self,
        *,
        vsi_map: np.ndarray | None = None,
        inplace: bool = True,
        similarity_kwargs: dict | None = None,
        heterotypic_overlap_kwargs: dict | None = None,
        vsi_kwargs: dict | None = None,
    ):
        """
        Run volume-layer (vl) metrics.

        Convenience wrapper around segtraq.vl functions via the instance facade `self.vl`.
        Runs, in order:

        1) similarity_top_bottom
        2) fraction_heterotypic_overlap
        3) vertical_signal_integrity_per_cell (only if `vsi_map` is provided)

        Parameters
        ----------
        vsi_map : np.ndarray or None, optional
            Precomputed 2D VSI map required for `vertical_signal_integrity_per_cell`.
            If None, VSI will be skipped.
        inplace : bool, default=True
            If True, metrics are written into `sdata.tables[tables_key].obs` by the
            underlying methods and this function returns None.
            If False, returns a dict of result DataFrames.
        similarity_kwargs : dict or None, optional
            Additional keyword arguments forwarded to :meth:`vl.similarity_top_bottom`.
        heterotypic_overlap_kwargs : dict or None, optional
            Additional keyword arguments forwarded to :meth:`vl.fraction_heterotypic_overlap`.
        vsi_kwargs : dict or None, optional
            Additional keyword arguments forwarded to :meth:`vl.vertical_signal_integrity_per_cell`.

        Returns
        -------
        None or dict[str, object]
            If `inplace=True`, returns None.

            If `inplace=False`, returns a dict with keys:

            - "similarity_top_bottom": pd.DataFrame
            - "fraction_heterotypic_overlap": pd.DataFrame
            - "vertical_signal_integrity_per_cell": pd.DataFrame   (only if vsi_map is not None)
        """
        assert self.points_z_key is not None, (
            "Cannot run volume metrics for 2D data: `points_z_key` is None. "
            "If available, define the column for z-coordinate of transcripts when initializing SegTraQ."
        )

        sim = self.vl.similarity_top_bottom(
            inplace=inplace,
            **(similarity_kwargs or {}),
        )

        het = self.vl.fraction_heterotypic_overlap(
            inplace=inplace,
            **(heterotypic_overlap_kwargs or {}),
        )

        vsi = None
        if vsi_map is not None:
            vsi = self.vl.vertical_signal_integrity_per_cell(
                vsi_map=vsi_map,
                inplace=inplace,
                **(vsi_kwargs or {}),
            )

        if inplace:
            return None

        out = {
            "similarity_top_bottom": sim,
            "fraction_heterotypic_overlap": het,
        }
        if vsi_map is not None:
            out["vertical_signal_integrity_per_cell"] = vsi

        return out

    def run_clustering_stability(
        self,
        key_prefix: str = "leiden_subset",
        use_hvg: bool = False,
        inplace: bool = True,
        connectedness_kwargs: dict | None = None,
        silhouette_kwargs: dict | None = None,
        purity_kwargs: dict | None = None,
        ari_kwargs: dict | None = None,
    ):
        """
        Run clustering-stability metrics.

        This method is a convenience wrapper around the clustering-stability (cs)
        functions. It runs, in order:

        1) cluster connectedness
        2) silhouette score
        3) purity (subset stability)
        4) ARI (subset stability)

        Only parameters shared by all four computations are exposed explicitly.
        All other parameters are provided via method-specific ``*_kwargs`` dictionaries.

        Parameters
        ----------
        key_prefix : str, default="leiden_subset"
            Prefix for Leiden clustering labels written to `.obs` by the underlying
            methods (where applicable).
        use_hvg: bool, optional
            Whether to use highly variable genes (HVGs) for PCA. By default False.
        inplace : bool, default=True
            If True, metrics are written to `sdata.tables["table"].uns` by the
            underlying methods and this function returns None. If False, the
            computed metrics are returned as a dictionary.
        connectedness_kwargs : dict or None, optional
            Additonal keyword arguments forwarded to :meth:`cs.compute_cluster_connectedness`.
        silhouette_kwargs : dict or None, optional
            Additonal keyword arguments forwarded to :meth:`cs.compute_silhouette_score`.
        purity_kwargs : dict or None, optional
            Additonal keyword arguments forwarded to :meth:`cs.compute_purity`.
        ari_kwargs : dict or None, optional
            Additonal keyword arguments forwarded to :meth:`cs.compute_ari`.

        Returns
        -------
        None or dict
            If `inplace=True`, returns None.
            If `inplace=False`, returns a dict with keys:

            - ``"cluster_connectedness"`` : float
            - ``"silhouette_score"`` : float
            - ``"mean_purity"`` : float
            - ``"mean_ari"`` : float
        """
        cc = self.cs.cluster_connectedness(
            key_prefix=key_prefix,
            use_hvg=use_hvg,
            inplace=inplace,
            **(connectedness_kwargs or {}),
        )

        sil = self.cs.silhouette_score(
            key_prefix=key_prefix,
            use_hvg=use_hvg,
            inplace=inplace,
            **(silhouette_kwargs or {}),
        )

        purity = self.cs.purity(
            key_prefix=key_prefix,
            use_hvg=use_hvg,
            inplace=inplace,
            **(purity_kwargs or {}),
        )

        ari = self.cs.adjusted_rand_index(
            key_prefix=key_prefix,
            use_hvg=use_hvg,
            inplace=inplace,
            **(ari_kwargs or {}),
        )

        if inplace:
            return None

        return {
            "cluster_connectedness": cc,
            "silhouette_score": sil,
            "mean_purity": purity,
            "mean_ari": ari,
        }

    def run_supervised(
        self,
        *,
        markers: dict[str, dict[str, list[str]]],
        cell_type_key: str = "transferred_cell_type",
        layer: str | None = None,
        inplace: bool = True,
        # per-metric parameters (optional)
        purity_kwargs: dict | None = None,
        contamination_kwargs: dict | None = None,
        mecr_kwargs: dict | None = None,
    ):
        """
        Run supervised (sp) metrics.

        Convenience wrapper around supervised marker-based QC metrics. Runs, in order:

        1) marker_purity (per-cell precision/recall/F1, neighborhood-aware negatives)
        2) neighbor_contamination (per-cell + directed type-type matrices)
        3) mutually_exclusive_coexpression_rate (MECR)

        Only parameters shared by all computations are exposed explicitly. All other
        parameters are forwarded via method-specific ``*_kwargs`` dictionaries.

        Parameters
        ----------
        markers : dict
            {cell_type: {"positive": list[str], "negative": list[str]}}.
        cell_type_key : str, default="transferred_cell_type"
            Column in the AnnData `.obs` with cell-type labels.
        layer : str | None, optional
            Layer containing count data. If `None`, `adata.X` is used if it looks
            like counts, otherwise `adata.layers["counts"]` is used if available.
            If a layer is specified, it must exist and contain count-like values.
        inplace : bool, default=True
            If True, writes results into `.obs` / `.uns` / `.uns[...]` as implemented
            by the underlying functions and returns None.
            If False, returns all results as a dict.
        purity_kwargs : dict or None, optional
            Extra args for :meth:`sp.marker_purity`.
            (e.g. use_quantiles=..., weight_cont=..., require_neighbor_expression=..., neighbors_key=...)
        contamination_kwargs : dict or None, optional
            Extra args for :meth:`sp.neighbor_contamination`.
            (e.g. require_neighbor_expression=..., neighbors_key=..., uns_key=..., uns_key_binary=...)
        mecr_kwargs : dict or None, optional
            Extra args for :meth:`sp.mutually_exclusive_coexpression_rate`.
            (e.g. pseudocount=...)

        Returns
        -------
        None or dict
            If ``inplace=True``, returns None.
            If ``inplace=False``, returns a dict with keys:
            - ``"marker_purity"`` (pd.DataFrame)
            - ``"neighbor_contamination"`` (dict with per-cell + matrices)
            - ``"mutually_exclusive_coexpression_rate"`` (pd.DataFrame)
        """
        purity_kwargs = {} if purity_kwargs is None else dict(purity_kwargs)
        contamination_kwargs = {} if contamination_kwargs is None else dict(contamination_kwargs)
        mecr_kwargs = {} if mecr_kwargs is None else dict(mecr_kwargs)

        # Respect runner-level inplace unless explicitly overridden per-metric
        purity_inplace = purity_kwargs.pop("inplace", inplace)
        cont_inplace = contamination_kwargs.pop("inplace", inplace)
        mecr_inplace = mecr_kwargs.pop("inplace", inplace)

        # 1) Marker purity
        purity_df = self.sp.marker_purity(
            cell_type_key=cell_type_key,
            layer=layer,
            markers=markers,
            inplace=purity_inplace,
            **purity_kwargs,
        )

        # 2) Neighbor contamination
        per_cell_cont_df, cont_mat_df, cont_bin_df = self.sp.neighbor_contamination(
            cell_type_key=cell_type_key,
            layer=layer,
            markers=markers,
            inplace=cont_inplace,
            **contamination_kwargs,
        )

        # 3) MECR
        mecr_df = self.sp.mutually_exclusive_coexpression_rate(
            markers=markers,
            layer=layer,
            inplace=mecr_inplace,
            **mecr_kwargs,
        )

        if inplace:
            return None

        return {
            "marker_purity": purity_df,
            "neighbor_contamination": {
                "per_cell": per_cell_cont_df,
                "matrix": cont_mat_df,
                "binary_matrix": cont_bin_df,
            },
            "mutually_exclusive_coexpression_rate": mecr_df,
        }

    def run_point_statistics(
        self,
        genes: str | list[str] | None = None,
        cell_type_key: str = "transferred_cell_type",
        cell_type_query: str | list[str] | None = None,
        inplace: bool = True,
        *,
        # per-metric parameters (optional)
        centroid_kwargs: dict | None = None,
        membrane_kwargs: dict | None = None,
        skew_kwargs: dict | None = None,
        compartments_kwargs: dict | None = None,
    ):
        """
        Run point-statistics (ps) metrics.

        Convenience wrapper around point-level spatial statistics. Applies shared
        transcript and cell filtering (by gene(s) and cell type) and runs, in order:

        1) percentage of transcripts in compartments (nucleus overlap, cytoplasm, outside)
        2) distance to centroid (cell or nucleus)
        3) distance to membrane (cell or nucleus)
        4) membrane-distance skewness

        Only parameters shared by all computations are exposed explicitly. All other
        parameters are forwarded via method-specific ``*_kwargs`` dictionaries.

        Parameters
        ----------
        genes : str | list[str] | None, optional
            Gene(s) to include. If None, all genes are used.
        cell_type_key : str, default="transferred_cell_type"
            Cell-type annotation key in `sdata.tables[...].obs`.
        cell_type_query : str | list[str] | None, optional
            Restrict computations to cells matching these label(s).
        inplace : bool, default=True
            If True, results are merged into `.obs` and None is returned.
            If False, per-metric results are returned.

        centroid_kwargs : dict or None, optional
            Extra arguments for :meth:`ps.distance_to_centroid`.
        membrane_kwargs : dict or None, optional
            Extra arguments for :meth:`ps.distance_to_membrane`.
        skew_kwargs : dict or None, optional
            Extra arguments for :meth:`ps.membrane_distance_skewness`.
        compartments_kwargs : dict or None, optional
            Extra arguments for :meth:`ps.percentage_transcripts_in_compartments`.

        Returns
        -------
        None or dict
            If ``inplace=True``, returns None.
            If ``inplace=False``, returns a dict with keys:
            - ``"percentage_transcripts_in_compartments"``
            - ``"distance_to_centroid"``
            - ``"distance_to_membrane"``
            - ``"membrane_distance_skewness"``
        """
        common = dict(
            genes=genes,
            cell_type_key=cell_type_key,
            cell_type_query=cell_type_query,
            inplace=inplace,
        )

        centroid_kwargs = {} if centroid_kwargs is None else dict(centroid_kwargs)
        membrane_kwargs = {} if membrane_kwargs is None else dict(membrane_kwargs)
        skew_kwargs = {} if skew_kwargs is None else dict(skew_kwargs)
        compartments_kwargs = {} if compartments_kwargs is None else dict(compartments_kwargs)

        # % compartments
        perc_cp_df = self.ps.percentage_transcripts_in_compartments(
            **common,
            **compartments_kwargs,
        )

        # mean-to-centroid distance
        cmd_df = self.ps.distance_to_centroid(
            **common,
            **centroid_kwargs,
        )

        # mean distance to membrane
        dtm_df = self.ps.distance_to_membrane(
            **common,
            **membrane_kwargs,
        )

        # skewness of distances-to-membrane
        mb_skw = self.ps.membrane_distance_skewness(
            **common,
            **skew_kwargs,
        )

        if inplace:
            return None

        return {
            "percentage_transcripts_in_compartments": perc_cp_df,
            "distance_to_centroid": cmd_df,
            "distance_to_membrane": dtm_df,
            "membrane_distance_skewness": mb_skw,
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
        use_hvg: bool = False,
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
        use_hvg: bool, optional
            Whether to use highly variable genes (HVGs) for PCA. By default False.
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
            use_hvg=use_hvg,
            inplace=inplace,
        )

        return None if inplace else result

    run_label_transfer.__doc__ = _run_label_transfer.__doc__

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
            sdata = sd.deepcopy(self.sdata)
        else:
            sdata = self.sdata

        assert adata.n_obs > 0, "Filtering removed all cells; no cells remain after filtering."

        sdata.tables[self.tables_key] = adata
        # sdata = sd.match_sdata_to_table(sdata, "table")
        # SpatialData currently only allows syncing one layer to tables, will be fixed in future release.
        # For now, we only filter the cells in the table.

        if inplace:
            self.sdata = sdata
            return None
        return sdata

    filter_cells.__doc__ = filter_cells.__doc__

    def filter_control_and_low_quality_transcripts(
        self,
        min_qv: float = 20.0,
        control_genes: tuple | list = (),
        recompute_expression: bool = False,
        inplace: bool = True,
    ):
        """
        Filter control and low-quality transcripts from the SpatialData object.

        Parameters
        ----------
        min_qv : float, default=20.0
            Minimum quality value (QV) threshold for transcripts to be retained.
        control_genes : tuple or list, optional
            List of gene name prefixes indicating control genes to be filtered out.
        recompute_expression : bool, default=False
            If True, recomputes the per-cell expression matrix after filtering transcripts.
        inplace : bool, default=True
            If True, modifies `self.sdata` in place.
            If False, returns a new SpatialData object with filtered transcripts.

        Returns
        -------
        None or SpatialData
            - If `inplace=True`: returns None after modifying `self.sdata`.
            - If `inplace=False`: returns a new SpatialData object with filtered transcripts.
        """
        _filter_control_and_low_quality_transcripts(
            sdata=self.sdata,
            min_qv=min_qv,
            control_genes=control_genes,
            points_key=self.points_key,
            points_gene_key=self.points_gene_key,
            points_cell_id_key=self.points_cell_id_key,
            tables_key=self.tables_key,
            recompute_expression=recompute_expression,
            inplace=inplace,
        )

    filter_control_and_low_quality_transcripts.__doc__ = _filter_control_and_low_quality_transcripts.__doc__


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
            points_background_id=self._p.points_background_id,
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
            points_background_id=self._p.points_background_id,
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
            points_background_id=self._p.points_background_id,
            inplace=inplace,
        )

    transcripts_per_cell.__doc__ = bl.transcripts_per_cell.__doc__

    def morphological_features(self, features_to_compute: list | None = None, n_jobs: int = 1, inplace: bool = True):
        return bl.morphological_features(
            sdata=self._p.sdata,
            tables_cell_id_key=self._p.tables_cell_id_key,
            tables_centroid_x_key=self._p.tables_centroid_x_key,
            tables_centroid_y_key=self._p.tables_centroid_y_key,
            shapes_key=self._p.shapes_key,
            features_to_compute=features_to_compute,
            n_jobs=n_jobs,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    morphological_features.__doc__ = bl.morphological_features.__doc__

    def transcript_density(self, inplace: bool = True):
        return bl.transcript_density(
            sdata=self._p.sdata,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            tables_area_key=self._p.tables_area_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_background_id=self._p.points_background_id,
            inplace=inplace,
        )

    transcript_density.__doc__ = bl.transcript_density.__doc__
    
    def image_features(self, features=("mean", "std", "median", "min", "max"), channel_names=None, inplace: bool = True):
        return bl.image_features(
            sdata=self._p.sdata,
            images_key=self._p.images_key,
            shapes_key=self._p.shapes_key,
            channel_names=channel_names,
            features=features,
            shapes_cell_id_key=self._p.shapes_cell_id_key,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            inplace=inplace
        )
        
    image_features.__doc__ = bl.image_features.__doc__


class _RSFacade:
    """
    Bound region-similarity (rs) metrics interface for a SegTraQ instance.
    Methods use the parent's `sdata` and configured keys.
    No per-call overrides are allowed.
    """

    def __init__(self, parent: "SegTraQ") -> None:
        self._p = parent

    def match_nuclei_to_cells(
        self,
        select_by: str = "nucleus_fraction",
        min_intersection_area: float = 0.0,
        n_jobs: int = -1,
        inplace: bool = True,
    ):
        return rs.match_nuclei_to_cells(
            sdata=self._p.sdata,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_key=self._p.shapes_key,
            nucleus_shapes_key=self._p.nucleus_shapes_key,
            select_by=select_by,
            min_intersection_area=min_intersection_area,
            n_jobs=n_jobs,
            inplace=inplace,
        )

    match_nuclei_to_cells.__doc__ = rs.match_nuclei_to_cells.__doc__

    def similarity_nucleus_cell(
        self,
        min_transcripts: int = 10,
        min_genes: int = 5,
        metric: str = "cosine_sim",
        select_by: str = "nucleus_fraction",
        min_intersection_area: float = 0.0,
        n_jobs: int = -1,
        inplace: bool = True,
    ):
        return rs.similarity_nucleus_cell(
            sdata=self._p.sdata,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_key=self._p.shapes_key,
            nucleus_shapes_key=self._p.nucleus_shapes_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_background_id=self._p.points_background_id,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            points_gene_key=self._p.points_gene_key,
            min_transcripts=min_transcripts,
            min_genes=min_genes,
            metric=metric,
            select_by=select_by,
            min_intersection_area=min_intersection_area,
            n_jobs=n_jobs,
            inplace=inplace,
        )

    similarity_nucleus_cell.__doc__ = rs.similarity_nucleus_cell.__doc__

    def similarity_nucleus_cytoplasm(
        self,
        min_transcripts: int = 10,
        min_genes: int = 5,
        metric: str = "cosine_sim",
        scale: float = 1e4,
        select_by: str = "nucleus_fraction",
        min_intersection_area: float = 0.0,
        n_jobs: int = -1,
        inplace: bool = True,
    ):
        return rs.similarity_nucleus_cytoplasm(
            sdata=self._p.sdata,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_key=self._p.shapes_key,
            nucleus_shapes_key=self._p.nucleus_shapes_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_background_id=self._p.points_background_id,
            points_gene_key=self._p.points_gene_key,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            min_transcripts=min_transcripts,
            min_genes=min_genes,
            metric=metric,
            scale=scale,
            select_by=select_by,
            min_intersection_area=min_intersection_area,
            n_jobs=n_jobs,
            inplace=inplace,
        )

    similarity_nucleus_cytoplasm.__doc__ = rs.similarity_nucleus_cytoplasm.__doc__

    def similarity_border_neighborhood(
        self,
        erosion_fraction_of_radius: float = 0.2,
        neighborhood_radius_factor: float = 2.0,
        min_transcripts: int = 10,
        min_genes: int = 5,
        metric: str = "cosine_sim",
        inplace: bool = True,
    ):
        return rs.similarity_border_neighborhood(
            sdata=self._p.sdata,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_key=self._p.shapes_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_background_id=self._p.points_background_id,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            points_gene_key=self._p.points_gene_key,
            erosion_fraction_of_radius=erosion_fraction_of_radius,
            min_transcripts=min_transcripts,
            min_genes=min_genes,
            neighborhood_radius_factor=neighborhood_radius_factor,
            metric=metric,
            inplace=inplace,
        )

    similarity_border_neighborhood.__doc__ = rs.similarity_border_neighborhood.__doc__

    # function for debugging / exploration
    # this function will not be highlighted in the main docs
    def get_genes_in_compartment(
        self,
        cell,
        compartment,
        scale: float = 1e4,
        erosion_fraction_of_radius: float = 0.2,
        neighborhood_radius_factor: float = 2.0,
    ):
        return rs.get_genes_in_compartment(
            cell=cell,
            compartment=compartment,
            sdata=self._p.sdata,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_key=self._p.shapes_key,
            nucleus_shapes_key=self._p.nucleus_shapes_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_background_id=self._p.points_background_id,
            points_gene_key=self._p.points_gene_key,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            scale=scale,
            erosion_fraction_of_radius=erosion_fraction_of_radius,
            neighborhood_radius_factor=neighborhood_radius_factor,
        )


class _SPFacade:
    """
    Bound supervised (sp) interface for a SegTraQ instance.
    Methods use the parent's `sdata` and configured `tables_key`.
    No per-call overrides are allowed.
    """

    def __init__(self, parent: "SegTraQ") -> None:
        self._p = parent

    def mutually_exclusive_coexpression_rate(
        self,
        markers: dict[str, dict[str, list[str]]],
        layer: str | None = None,
        pseudocount: float = 0.5,
        inplace: bool = True,
    ):
        return sp.mutually_exclusive_coexpression_rate(
            sdata=self._p.sdata,
            markers=markers,
            layer=layer,
            pseudocount=pseudocount,
            tables_key=self._p.tables_key,
            inplace=inplace,
        )

    mutually_exclusive_coexpression_rate.__doc__ = sp.mutually_exclusive_coexpression_rate.__doc__

    def marker_purity(
        self,
        cell_type_key: str,
        markers: dict[str, dict[str, list[str]]],
        layer: str | None = None,
        use_quantiles: bool = False,
        require_neighbor_expression: bool = True,
        weight_cont: float = 0.7,
        neighbors_key: str | None = "spatial_connectivities",
        inplace: bool = True,
    ):
        return sp.marker_purity(
            sdata=self._p.sdata,
            cell_type_key=cell_type_key,
            markers=markers,
            layer=layer,
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

    marker_purity.__doc__ = sp.marker_purity.__doc__

    def neighbor_contamination(
        self,
        cell_type_key: str,
        markers: dict[str, dict[str, list[str]]],
        layer: str | None = None,
        require_neighbor_expression: bool = True,
        neighbors_key: str | None = "spatial_connectivities",
        uns_key: str = "negative_marker_contamination",
        uns_key_binary: str = "negative_marker_contamination_binary",
        inplace: bool = True,
    ):
        return sp.neighbor_contamination(
            sdata=self._p.sdata,
            cell_type_key=cell_type_key,
            markers=markers,
            layer=layer,
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

    neighbor_contamination.__doc__ = sp.neighbor_contamination.__doc__


class _PSFacade:
    """
    Bound points-statistics (ps) interface for a SegTraQ instance.
    Methods use the parent's `sdata` and configured keys.
    No per-call overrides are allowed.
    """

    def __init__(self, parent: "SegTraQ") -> None:
        self._p = parent

    def percentage_transcripts_in_compartments(
        self,
        genes: str | list[str] = None,
        cell_type_key: str | None = "transferred_cell_type",
        cell_type_query: str | list[str] | None = None,
        select_by: Literal["iou", "nucleus_fraction"] = "nucleus_fraction",
        min_intersection_area: float = 0.0,
        n_jobs: int = 1,
        predicate: str = "intersects",
        inplace: bool = True,
    ):
        return ps.percentage_transcripts_in_compartments(
            sdata=self._p.sdata,
            genes=genes,
            cell_type_key=cell_type_key,
            cell_type_query=cell_type_query,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            shapes_key=self._p.shapes_key,
            nucleus_shapes_key=self._p.nucleus_shapes_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_background_id=self._p.points_background_id,
            points_gene_key=self._p.points_gene_key,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            select_by=select_by,
            min_intersection_area=min_intersection_area,
            n_jobs=n_jobs,
            predicate=predicate,
            inplace=inplace,
        )

    percentage_transcripts_in_compartments.__doc__ = ps.percentage_transcripts_in_compartments.__doc__

    def distance_to_centroid(
        self,
        genes: str | list[str] = None,
        cell_type_key: str | None = "transferred_cell_type",
        cell_type_query: str | list[str] | None = None,
        centroid_region: Literal["cell", "nucleus"] = "cell",
        restrict_to_within_boundary: bool = False,
        select_by: Literal["iou", "nucleus_fraction"] = "nucleus_fraction",
        min_intersection_area: float = 0.0,
        n_jobs: int = 1,
        inplace: bool = True,
    ):
        return ps.distance_to_centroid(
            sdata=self._p.sdata,
            genes=genes,
            cell_type_key=cell_type_key,
            cell_type_query=cell_type_query,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            tables_area_key=self._p.tables_area_key,
            points_gene_key=self._p.points_gene_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_background_id=self._p.points_background_id,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            shapes_key=self._p.shapes_key,
            nucleus_shapes_key=self._p.nucleus_shapes_key,
            centroid_region=centroid_region,
            restrict_to_within_boundary=restrict_to_within_boundary,
            select_by=select_by,
            min_intersection_area=min_intersection_area,
            n_jobs=n_jobs,
            inplace=inplace,
        )

    distance_to_centroid.__doc__ = ps.distance_to_centroid.__doc__

    def distance_to_membrane(
        self,
        genes: str | list[str] | None = None,
        cell_type_key: str | None = "transferred_cell_type",
        cell_type_query: str | list[str] | None = None,
        restrict_to_within_boundary: bool = False,
        membrane_region: Literal["cell", "nucleus"] = "cell",
        select_by: Literal["iou", "nucleus_fraction"] = "nucleus_fraction",
        min_intersection_area: float = 0.0,
        n_jobs: int = 1,
        signed: bool = True,
        inverse_score: bool = True,
        eps: float = 1e-6,
        inplace: bool = True,
    ):
        return ps.distance_to_membrane(
            sdata=self._p.sdata,
            genes=genes,
            cell_type_key=cell_type_key,
            cell_type_query=cell_type_query,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            tables_area_key=self._p.tables_area_key,
            points_gene_key=self._p.points_gene_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_background_id=self._p.points_background_id,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            shapes_key=self._p.shapes_key,
            nucleus_shapes_key=self._p.nucleus_shapes_key,
            membrane_region=membrane_region,
            restrict_to_within_boundary=restrict_to_within_boundary,
            select_by=select_by,
            min_intersection_area=min_intersection_area,
            n_jobs=n_jobs,
            signed=signed,
            inverse_score=inverse_score,
            eps=eps,
            inplace=inplace,
        )

    distance_to_membrane.__doc__ = ps.distance_to_membrane.__doc__

    def membrane_distance_skewness(
        self,
        genes: str | list[str] | None = None,
        cell_type_key: str = "transferred_cell_type",
        cell_type_query: str | list[str] | None = None,
        min_transcripts: int = 5,
        inplace: bool = True,
    ):
        return ps.membrane_distance_skewness(
            sdata=self._p.sdata,
            genes=genes,
            cell_type_key=cell_type_key,
            cell_type_query=cell_type_query,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            points_gene_key=self._p.points_gene_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_background_id=self._p.points_background_id,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            shapes_key=self._p.shapes_key,
            min_transcripts=min_transcripts,
            inplace=inplace,
        )

    membrane_distance_skewness.__doc__ = ps.membrane_distance_skewness.__doc__


class _CSFacade:
    """
    Thin facade over segtraq.cs bound to a SegTraQ instance.
    Methods use the parent's sdata and configured keys exclusively.
    No per-call overrides are allowed.
    """

    def __init__(self, parent: "SegTraQ") -> None:
        self._p = parent

    def silhouette_score(
        self,
        resolution: float | list[float] = (0.6, 0.8, 1.0),
        metric: str = "euclidean",
        key_prefix: str = "leiden_subset",
        random_state: int = 42,
        cell_type_key: str | None = None,
        use_hvg: bool = False,
        inplace: bool = True,
    ) -> float:
        return cs.silhouette_score(
            self._p.sdata,
            resolution=resolution,
            metric=metric,
            tables_key=self._p.tables_key,
            key_prefix=key_prefix,
            random_state=random_state,
            cell_type_key=cell_type_key,
            use_hvg=use_hvg,
            inplace=inplace,
        )

    silhouette_score.__doc__ = cs.silhouette_score.__doc__

    def purity(
        self,
        resolution: float = 1.0,
        frac_cells_subset: float = 0.63,
        key_prefix: str = "leiden_subset",
        use_hvg: bool = False,
        representation: str | None = None,
        inplace: bool = True,
    ) -> float:
        return cs.purity(
            self._p.sdata,
            resolution=resolution,
            frac_cells_subset=frac_cells_subset,
            tables_key=self._p.tables_key,
            key_prefix=key_prefix,
            use_hvg=use_hvg,
            representation=representation,
            inplace=inplace,
        )

    purity.__doc__ = cs.purity.__doc__

    def adjusted_rand_index(
        self,
        resolution: float = 1.0,
        frac_cells_subset: float = 0.63,
        key_prefix: str = "leiden_subset",
        use_hvg: bool = False,
        representation: str | None = None,
        inplace: bool = True,
    ) -> float:
        return cs.adjusted_rand_index(
            self._p.sdata,
            resolution=resolution,
            frac_cells_subset=frac_cells_subset,
            key_prefix=key_prefix,
            tables_key=self._p.tables_key,
            use_hvg=use_hvg,
            representation=representation,
            inplace=inplace,
        )

    adjusted_rand_index.__doc__ = cs.adjusted_rand_index.__doc__

    def cluster_connectedness(
        self,
        resolution: float | list[float] = (0.6, 0.8, 1.0),
        use_weights: bool = False,
        key_prefix: str = "leiden_subset",
        random_state: int = 42,
        cell_type_key: str | None = None,
        use_hvg: bool = False,
        inplace: bool = True,
    ):
        return cs.cluster_connectedness(
            sdata=self._p.sdata,
            resolution=resolution,
            use_weights=use_weights,
            key_prefix=key_prefix,
            tables_key=self._p.tables_key,
            random_state=random_state,
            cell_type_key=cell_type_key,
            use_hvg=use_hvg,
            inplace=inplace,
        )

    cluster_connectedness.__doc__ = cs.cluster_connectedness.__doc__


class _VLFacade:
    """
    Thin facade over segtraq.vl bound to a SegTraQ instance.
    Methods use the parent's sdata and configured keys exclusively.
    No per-call overrides are allowed.
    """

    def __init__(self, parent: "SegTraQ") -> None:
        self._p = parent

    def similarity_top_bottom(
        self,
        correct_z_drift: bool = True,
        max_points: int = 1_000_000,
        seed: int | None = 0,
        q: float = 0.30,
        scale: float = 1e4,
        normalization: str | None = "log",
        min_genes: int = 5,
        min_transcripts: int = 10,
        inplace: bool = True,
    ):
        return vl.similarity_top_bottom(
            self._p.sdata,
            tables_key=self._p.tables_key,
            tables_cell_id_key=self._p.tables_cell_id_key,
            points_key=self._p.points_key,
            points_cell_id_key=self._p.points_cell_id_key,
            points_background_id=self._p.points_background_id,
            points_gene_key=self._p.points_gene_key,
            points_x_key=self._p.points_x_key,
            points_y_key=self._p.points_y_key,
            points_z_key=self._p.points_z_key,
            correct_z_drift=correct_z_drift,
            max_points=max_points,
            seed=seed,
            q=q,
            normalization=normalization,
            scale=scale,
            min_genes=min_genes,
            min_transcripts=min_transcripts,
            inplace=inplace,
        )

    similarity_top_bottom.__doc__ = vl.similarity_top_bottom.__doc__

    def fraction_heterotypic_overlap(
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
        return vl.fraction_heterotypic_overlap(
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

    fraction_heterotypic_overlap.__doc__ = vl.fraction_heterotypic_overlap.__doc__

    def vertical_signal_integrity_per_cell(
        self,
        vsi_map: np.ndarray,
        inplace: bool = True,
    ):
        return vl.vertical_signal_integrity_per_cell(
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
            inplace=inplace,
        )

    vertical_signal_integrity_per_cell.__doc__ = vl.vertical_signal_integrity_per_cell.__doc__
