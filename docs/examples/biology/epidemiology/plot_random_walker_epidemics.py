"""
# Random-walker epidemics on a grid and on a small world

Random walkers among whom one sick walker starts a deadly epidemic, on a grid or on a network.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'random_walker_epidemics'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
