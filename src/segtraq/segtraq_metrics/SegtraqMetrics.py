import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
from pathlib import Path
from scipy.spatial.distance import cdist
from typing import Dict, Optional, Union
from .. import fs, sp, nc, bl, cs, pl
from ..utils import run_label_transfer

class SegTraqer:
    def __init__(self, 
                 sdata,
                 adata_ref,
                 ref_celltype_key: str = "celltype",
                 filter_tx_min: float = 10.0,
                 filter_tx_max: float = 2000.0,
                 filter_gn_min: float = 5.0,
                 filter_gn_max: float = np.inf,
                 palette: Optional[Dict[str, str]] = None,
                 out_path: Union[str, Path] = None,
                 shape_key: str | list[str] = "cell_boundaries",
                 shape_key_nc: str = "cell_boundaries",
                 label_key: str | list[str] = "cell_labels",
                 label_key_nc: str = "cell_labels",
                 points_key: str = "transcripts",
                 table_key: str = "table",
                 cell_key_points: str = "cell_id",
                 cell_key_shapes: str = "cell_id",
                 cell_key_tables: str = "cell_id",
                 gene_key: str = "feature_name",
                 data_key: str | None = None,
                 area_key: str = "volume",
                 background_cell_id: str | int = "UNASSIGNED",
                 nuc_shape_key: str = "nucleus_boundaries",
                 x_coordinate: str = "x",
                 y_coordinate: str = "y",
                 validate: bool = True):

        self.sdata = sdata
        self.adata_ref = adata_ref
        self.ref_celltype_key = ref_celltype_key 
        self.filter_tx_min = filter_tx_min
        self.filter_tx_max = filter_tx_max
        self.filter_gn_min = filter_gn_min
        self.filter_gn_max = filter_gn_max
        self.palette = palette
        self.out_path = out_path
        self.shape_keys = (shape_key if isinstance(shape_key, (list, tuple)) else [shape_key])
        self.shape_key_nc = shape_key_nc
        self.label_keys = (label_key if isinstance(label_key, (list, tuple)) else [label_key])
        self.label_keys_nc = label_key_nc
        self.points_key = points_key
        self.table_key = table_key
        self.cell_key_points = cell_key_points
        self.cell_key_shapes = cell_key_shapes
        self.cell_key_tables = cell_key_tables
        self.gene_key = gene_key
        self.data_key = data_key
        self.area_key = area_key
        self.background_cell_id = background_cell_id
        self.nuc_shape_key = nuc_shape_key
        self.x = x_coordinate
        self.y = y_coordinate

        if validate:
            fs.validate_spatialdata(
                sdata,
                shape_key=self.shape_keys,
                label_key=self.label_keys_nc,
                points_key=self.points_key,
                table_key=self.table_key,
                cell_key_points=self.cell_key_points,
                cell_key_shapes=self.cell_key_shapes,
                cell_key_tables=self.cell_key_tables,
                data_key=self.data_key,
                background_cell_id=self.background_cell_id,
            )

    def run_baseline(self, inplace=True):
        tbl = self.sdata.tables[self.table_key]
        gpc = bl.genes_per_cell(self.sdata, self.points_key, self.cell_key_tables, self.gene_key).set_index(self.cell_key_tables)
        tpc = bl.transcripts_per_cell(self.sdata, self.points_key, self.cell_key_tables).set_index(self.cell_key_tables)
        dens = bl.transcript_density(self.sdata, self.table_key, self.points_key, self.cell_key_tables, self.area_key).set_index(self.cell_key_tables)

        summary = dict(
            num_cells=bl.num_cells(self.sdata),
            num_genes=bl.num_genes(self.sdata, self.points_key, self.gene_key),
            num_transcripts=bl.num_transcripts(self.sdata, self.points_key),
            perc_unassigned_transcripts=bl.perc_unassigned_transcripts(
                self.sdata, self.points_key, self.cell_key_tables, self.background_cell_id
            )
        )

        if inplace:
            to_join = gpc.join(tpc, how="outer").join(dens, how="outer")
            tbl.obs = tbl.obs.merge(
                to_join,
                how="left",
                left_on="cell_id",  
                right_on="cell_id", 
            )
            tbl.uns.setdefault("segtraq", {}).setdefault("bl", {})["summary"] = summary
            return None
        else:
            return {
                "summary": summary,
                "genes_per_cell": gpc,
                "transcripts_per_cell": tpc,
                "transcript_density": dens,
            }
        

    
        
    def run_annotation(self,
                       inplace: bool = True):
        return run_label_transfer(
            sdata=self.sdata,
            adata_reference=self.adata_ref,
            celltype_key=self.ref_celltype_key,
            table_key=self.table_key,
            tx_min=self.filter_tx_min,
            tx_max=self.filter_tx_max,
            gn_min=self.filter_gn_min,
            gn_max=self.filter_gn_max,
            inplace=inplace
        )
    
    def run_umap(self, inplace: bool = True):
        adata = self.sdata.tables[self.table_key]

        if inplace:
            sc.pp.pca(adata)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            return None
        else:
            adata_copy = adata.copy()
            sc.pp.pca(adata_copy)
            sc.pp.neighbors(adata_copy)
            sc.tl.umap(adata_copy)
            return adata_copy


    def run_nuclear_correlation(
        self,
        inplace: bool = True,

    ):
        tbl = self.sdata.tables[self.table_key]

        # 1) IoU between cell and its best-matching nucleus
        ious = nc.nuclear_correlation.compute_cell_nuc_ious(
            sdata=self.sdata,
            cell_id_key_shape= self.cell_key_shapes,
            cell_shape_key=self.shape_key_nc,
            nuc_shape_key=self.nuc_shape_key,
            n_jobs=-1,
            use_progress=True,
        ).set_index(self.cell_key_tables)

        if inplace:
            tbl.obs = tbl.obs.merge(
                ious.reset_index(),
                how="left",
                left_on=self.cell_key_tables,
                right_on=self.cell_key_tables,
            )

        print("IoU computed.")

        # 2) Cell–nucleus correlation per cell
        cell_nuc_corr = nc.nuclear_correlation.compute_cell_nuc_correlation(
            sdata=self.sdata,
            table_key=self.table_key,
            cell_id_key=self.cell_key_tables,
            transcripts_key=self.points_key,
            nucleus_by=self.nuc_shape_key,
            feature_column=self.gene_key,
            x_coordinate=self.x,
            y_coordinate=self.y,
            cell_shape_key=self.shape_key_nc
        )
        cell_nuc_corr = cell_nuc_corr.set_index(self.cell_key_tables).rename(
        columns={
            "correlation": "corr_nc_cell",
        }
        )
        print("Cell-nuc correlation computed.")

        # 3) Correlation between nuclear-overlap part of cell vs rest of cell 
        parts_corr = nc.nuclear_correlation.compute_correlation_between_parts(
            sdata=self.sdata,
            table_key=self.table_key,
            cell_id_key_shape = self.cell_key_shapes,
            cell_shape_key=self.shape_key_nc,
            nuc_shape_key=self.nuc_shape_key,
            transcripts_key=self.points_key,
            feature_column=self.gene_key,
            x_coordinate=self.x,
            y_coordinate=self.y,
        ).set_index(self.cell_key_tables).rename(
        columns={
            "correlation_parts": "corr_cell_parts",
        }
        )
        print("Correlation between parts correlation computed.")

        if inplace:
            to_join = (
                cell_nuc_corr.loc[:, ["corr_nc_cell"]]
                    .join(parts_corr.loc[:, ["corr_cell_parts"]], how="outer")
                    .reset_index()
            )
            tbl.obs = tbl.obs.merge(
                to_join,
                how="left",
                left_on=self.cell_key_tables,
                right_on=self.cell_key_tables,
            )
            return None
        else:
            return {
                "cell_nuc_correlation": cell_nuc_corr.reset_index(),
                "ious": ious.reset_index(),
                "parts_correlation": parts_corr.reset_index(),
            }
        
    def run_clustering_stability(
        self,
        inplace: bool = True,
        resolution: float | tuple[float, ...] = (0.6, 0.8, 1.0),
        key_prefix: str = "leiden_subset",
        n_genes_subset: int = 100,
        metric: str = "euclidean",
        ncomps: int = 30,
        random_state: int = 42,
    ):
        rmsd = cs.clustering_stability.compute_rmsd(
            self.sdata, resolution=resolution, key_prefix=key_prefix, random_state=random_state
        )
        print("RMSD computed.")
        sil = cs.clustering_stability.compute_silhouette_score(
            self.sdata, resolution=resolution, metric=metric, ncomps=ncomps,
            key_prefix=key_prefix, random_state=random_state
        )
        print("Silhouette Score computed.")
        ari = cs.clustering_stability.compute_ari(
            self.sdata, resolution=(resolution if isinstance(resolution, (int, float)) else 1.0),
            n_genes_subset=n_genes_subset, key_prefix=key_prefix
        )
        print("ARI computed.")
        purity = cs.clustering_stability.compute_purity(
            self.sdata, resolution=(resolution if isinstance(resolution, (int, float)) else 1.0),
            n_genes_subset=n_genes_subset, key_prefix=key_prefix
        )
        print("Purity computed.")
        result = {"rmsd": float(rmsd), "silhouette": float(sil), "ari": float(ari), "purity": float(purity)}

        tbl = self.sdata.tables[self.table_key]
        to_drop = [c for c in tbl.obs.columns if (c.startswith("leiden_subset_100") or c.startswith("leiden_subset_allgenes"))]
        tbl.obs.drop(columns=to_drop, inplace=True, errors="ignore")

        if inplace:
            tbl.uns.setdefault("segtraq", {}).setdefault("cs", {}).update(result)
            return None
        else:
            return result
        
    def run_supervised_spillover_metrics(
            self, 
            inplace: bool = True
    ):
        tbl = self.sdata.tables[self.table_key]
        common_genes = tbl.var_names[tbl.var_names.isin(self.adata_ref.var_names)]
        self.adata_ref = self.adata_ref[:, common_genes].copy()

        markers_dict = sp.find_markers_cellspa(self.adata_ref, self.ref_celltype_key)
        print("Identified positive and negative markers.")

        #mut_markers_dict = sp.find_mutually_exclusive_genes(self.adata_ref, markers_dict, self.ref_celltype_key) - this blows up memory - TODO find out how to store in sdata differently
        #print("Identified mutually exclusive markers.")

        #mecr = sp.compute_MECR(self.sdata, mut_markers_dict)
        #print("Computed mutually exclusive co-expression rate.")

        if "transferred_celltype" not in tbl.obs.columns:
            self.run_annotation()

        purity_results = sp.calculate_marker_purity(self.sdata, "transferred_celltype", markers_dict)
        print("Computed signal purity scores.")

        if inplace:
            tbl.obs = tbl.obs.merge(
                purity_results.reset_index(),
                how="left",
                left_on="cell_id",  
                right_on="cell_id", 
            )

            #mecr_flat = {f"{g1}__{g2}": float(v) for (g1, g2), v in mecr.items()}
            #tbl.uns.setdefault("segtraq", {})["sp"] = mecr_flat

        else:
            return purity_results#, mecr
        
    def run_rastering(
            self
    ):
        pl.save_mask_to_tiff(self.sdata, 
                        labels_keys=self.label_keys,
                        output_dir=self.out_path,
                        palette=self.palette,
                        unassigned_cell_id=self.background_cell_id)


    def run_all(
            self
    ):
        self.run_baseline()
        self.run_annotation()
        self.run_umap()
        #self.run_nuclear_correlation()
        self.run_clustering_stability()
        self.run_supervised_spillover_metrics()
        self.run_rastering()


    

