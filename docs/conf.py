# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SegTraQ"
copyright = "2025, Daria Lazic, Matthias Meyer-Bender, Martin Emons"
author = "Daria Lazic, Matthias Meyer-Bender, Martin Emons"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "myst_parser",
    "IPython.sphinxext.ipython_console_highlighting",
]

nbsphinx_execute = "never"  # notebooks arrive pre-executed
nbsphinx_execute_arguments = [
    "--InlineBackend.print_figure_kwargs={'bbox_inches': 'tight', 'transparent': True}",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "conf.py", "notebooks/*.py"]

nbsphinx_custom_formats = {
    ".py": ["jupytext.reads", {"fmt": "py:percent"}],
}

# Show both the class docstring and __init__ docstring
autoclass_content = "both"

# Hide type hints in the signatures; the types are documented in the docstrings instead
autodoc_typehints = "none"

# Turn the types in the numpydoc "Parameters"/"Returns" sections into cross-references,
# so they get rendered as highlighted (and, where possible, linked) code
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "sd.SpatialData": "~spatialdata.SpatialData",
    "SpatialData": "~spatialdata.SpatialData",
    "AnnData": "~anndata.AnnData",
    "ad.AnnData": "~anndata.AnnData",
    "np.ndarray": "~numpy.ndarray",
    "pd.DataFrame": "~pandas.DataFrame",
    "pd.Series": "~pandas.Series",
    "gpd.GeoDataFrame": "~geopandas.GeoDataFrame",
    "GeoDataFrame": "~geopandas.GeoDataFrame",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "spatialdata": ("https://spatialdata.scverse.org/en/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
}

templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = project
html_favicon = "_static/img/icon.png"
html_theme_options = {
    "home_page_in_toc": False,
    "navigation_with_keys": True,
    "logo": {
        "image_light": "_static/img/logo_light.png",
        "image_dark": "_static/img/logo_dark.png",
    },
}
# Enable Pygments syntax highlighting
highlight_language = "python"  # or 'none', 'bash', etc.
pygments_style = "default"  # or 'default', 'friendly', 'monokai', etc.
