"""
# Light-sensitive robots with sensory delay

Light-sensitive robots in a circular well, whose sensory delay decides whether they gather in the light.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'light_seeking_robots'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
