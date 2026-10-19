"""
# Bistability on a catalyst pellet

The surface concentration of a catalyst pellet, bistable through Langmuir-Hinshelwood kinetics.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'catalyst_pellet'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
