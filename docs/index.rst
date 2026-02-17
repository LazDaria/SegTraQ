.. SegTraQ documentation master file, created by
   sphinx-quickstart on Mon Aug  4 17:10:41 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SegTraQ documentation
=====================

SegTraQ (**Seg**mentation and **Tra**nscript Assignment **Q**uality Control) is a Python toolkit for quantitative and visual quality control of segmentation and transcript assignment in spatial omics data.

⚠️ Note: SegTraQ is under active development. 
Features, interfaces, and functionality may change in upcoming releases.
To install the latest development version, run `pip install git+https://github.com/LazDaria/SegTraQ`.

.. image:: _static/img/figure_1.png
   :width: 100%
   :align: center
   :alt: SegTraQ provides quality control metrics for segmentation of spatial transcriptomics data.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   modules/baseline
   modules/clustering_stability
   modules/region_similarity
   modules/supervised
   modules/point_statistics
   modules/volume
   modules/plotting
   modules/segtraq_class
   modules/utils

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:
   
   notebooks/io
   notebooks/baseline
   notebooks/clustering_stability
   notebooks/region_similarity
   notebooks/volume
   notebooks/supervised
   notebooks/point_statistics
   notebooks/plotting
   notebooks/10x_xenium_focus_simplified
   notebooks/10x_xenium_focus
   # notebooks/vizgen_merscope_focus
   # notebooks/cosmx_focus
