"""
# Active Ornstein-Uhlenbeck particles

Active Ornstein-Uhlenbeck particles in the plane, driven by exponentially correlated propulsion.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'active_ornstein_uhlenbeck_particles'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
