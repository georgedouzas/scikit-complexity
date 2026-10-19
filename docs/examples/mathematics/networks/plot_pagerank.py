"""
# PageRank iterations on networks

The PageRank iteration `r_i' = (1 - d) / n + d Σ_j A_ij r_j / k_j` on an undirected network.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'pagerank'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
