"""
# Bak-Tang-Wiesenfeld sandpile

The Bak-Tang-Wiesenfeld sandpile on a lattice with open edges, started from random stable heights.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'sandpile'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
