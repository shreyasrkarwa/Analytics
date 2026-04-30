import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'b2b-territory-optimization'
copyright = '2026, Shreyas Karwa'
author = 'Shreyas Karwa'
release = '0.1.3'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'numpydoc',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
