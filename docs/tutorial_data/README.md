# Tutorial data

The tutorials use small crops of the datasets analysed in the SegTraQ manuscript.
`prepare_tutorial_data.py` builds them from the outputs of the
[segtraq_analysis](https://github.com/LazDaria/segtraq_analysis) pipelines. It only reads
these outputs and writes everything to `--out`.

| File | Content | Source |
|---|---|---|
| `xenium_breast_crop.zarr` | Xenium human breast, Xenium segmentation, 450 µm crop (3× Fig. 4a) | GEO GSM9509160 |
| `proseg_breast_crop_raw.zarr` | same crop, Proseg 3 output as written by Proseg | Proseg on GSM9509160 |
| `proseg_breast_crop.zarr` | same crop, Proseg harmonised with the Xenium data (image, nuclei, measured coordinates, all transcripts) | segtraq_analysis chapter 4 |
| `visium_hd_ovarian_crop_raw.zarr` | Visium HD human ovarian cancer, 500 µm crop (2× Fig. 5b), as read by spatialdata-io (2 µm bins, Space Ranger cells) | 10x Genomics |
| `visium_hd_ovarian_crop.zarr` | same crop, cells with transcripts from `segtraq.bins_to_transcripts` | segtraq_analysis chapter 5 |
| `cosmx_lung_fov11.zarr` | CosMx lung adenocarcinoma, FOV 11, without SegTraQ metrics | GEO GSM9046088 |
| `reference_breast.h5ad`, `reference_ovarian.h5ad`, `reference_lung.h5ad` | scRNA-seq references, at most 500 cells per cell type, all genes | see segtraq_analysis chapters 3–5 |

The 36 CosMx FOVs with SegTraQ metrics are the chapter 3 archive of segtraq_analysis
(`segtraq_multisample_cosmx_tma1.zip`).

Crops are bounding-box queries: transcripts inside the square are kept, and cells whose
outline intersects the square are kept with their row in the table.

```bash
python prepare_tutorial_data.py --analysis-data DATA --out OUT xenium proseg visium cosmx references
python prepare_tutorial_data.py --analysis-data DATA --out OUT visium_raw   # ~256 GB memory
```
