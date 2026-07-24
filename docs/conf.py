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
    "sphinx.ext.napoleon",  # Optional but useful for Google/Numpy-style docstrings
    "nbsphinx",
    "myst_parser",
    "IPython.sphinxext.ipython_console_highlighting",
]

nbsphinx_custom_formats = {
    ".py": ["jupytext.reads", {"fmt": "py:percent"}],
}
nbsphinx_execute = "never"
nbsphinx_execute_arguments = [
    "--InlineBackend.print_figure_kwargs={'bbox_inches': 'tight', 'transparent': True}",
]

# Show both the class docstring and __init__ docstring
autoclass_content = "both"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "conf.py"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = project
html_favicon = "_static/img/icon.png"
html_theme_options = {
    "logo_only": True,
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
