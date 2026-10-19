"""
# Yeast cell-cycle network

The Boolean network of the budding yeast cell cycle of Li et al., eleven proteins and their regulation.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'yeast_cell_cycle'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
