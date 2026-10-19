"""
# Discrete-velocity Boltzmann gas of Carleman

The Carleman discrete-velocity Boltzmann equation on a periodic line, from a pulse of right movers.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'discrete_velocity_boltzmann'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
