"""
# Huygens clocks and pendulums on coupled supports

Pendulum clocks on supports joined by springs, which synchronize through the motion of the supports.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'huygens_clocks'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
