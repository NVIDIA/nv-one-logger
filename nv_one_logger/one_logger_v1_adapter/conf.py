import os
import sys

sys.path.insert(0, os.path.abspath("./src"))

project = "nv-one-logger-v1-adapter"
copyright = "2025, NVIDIA"
author = "NVIDIA"
release = "https://gitlab.com/pages/sphinx"
extensions = []
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
html_static_path = ["_static"]
