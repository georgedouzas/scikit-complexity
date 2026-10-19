"""
# Active Brownian particles

Active Brownian particles in the plane, which swim at a constant speed in a diffusing direction.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'active_brownian_particles'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
