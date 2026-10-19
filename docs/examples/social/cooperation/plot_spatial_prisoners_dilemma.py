"""
# Spatial prisoner's dilemma with best-neighbour and Fermi imitation

The spatial prisoner's dilemma of Nowak and May, with best-neighbour or Fermi imitation.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'spatial_prisoners_dilemma'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
