# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'Chunking for Pandas'
copyright = '2024, Jonathan Reich'
author = 'Jonathan Reich'
release = '0.1.2'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

# Templates
templates_path = ['_templates']
exclude_patterns = []

# HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static'] 