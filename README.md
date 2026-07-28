# SegTraQ

[![PyPI version](https://badge.fury.io/py/segtraq.svg)](https://badge.fury.io/py/segtraq)

> ⚠️ On July 28th, 2026, the history of this repository was rewritten to reduce its size. If you cloned the repo before this date and intend to contribute to this project, please create a fresh clone to avoid any conflicts!


> ⚠️ Note: SegTraQ is under active development. 
> Features, interfaces, and functionality may change in upcoming releases. 
> SegTraQ currently supports imaging-based spatial transcriptomics data only. 
> Support for sequencing-based spatial transcriptomics is under development and will be included in a future release.
> To install the latest development version, run `pip install git+https://github.com/LazDaria/SegTraQ`.

SegTraQ (**Seg**mentation and **Tra**nscript Assignment **Q**uality Control) is a Python toolkit for quantitative and visual quality control of segmentation and transcript assignment in spatial omics data.

## Getting Started
Please refer to the [documentation](https://lazdaria.github.io/SegTraQ) for details on the API and tutorials.

## Installation

To install `SegTraQ`, first create a python environment and install the package using 

```
pip install segtraq
```

The installation of the package should take less than a minute.

## System Requirements
### Hardware Requirements
`SegTraQ` requires only a standard computer with enough RAM to support the in-memory operations.

### Software Requirements
`SegTraQ` depends on the following packages:
```
scanpy
spatialdata
geopandas
rtree
rasterio
squidpy
anndata
ovrlpy
```