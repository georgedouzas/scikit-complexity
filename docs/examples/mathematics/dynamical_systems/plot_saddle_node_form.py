"""
# Saddle-node

The saddle-node normal form `x' = μ - x²`, whose two equilibria `±√μ` meet and vanish at μ = 0.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'saddle_node_form'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
