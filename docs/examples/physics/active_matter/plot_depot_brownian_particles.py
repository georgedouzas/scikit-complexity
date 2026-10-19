"""
# Depot-pumped Brownian particles

Brownian particles with energy depots, which start at rest and take up the speed `sqrt(q/gamma - c/d)`.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'depot_brownian_particles'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
