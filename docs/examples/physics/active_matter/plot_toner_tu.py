"""
# Toner-Tu hydrodynamics

The Toner-Tu hydrodynamics of a polar flock on a periodic lattice, from a disordered velocity field.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'toner_tu'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
