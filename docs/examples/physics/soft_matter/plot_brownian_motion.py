"""
# Inertial and overdamped Brownian motion

Free Brownian particles with inertia, `m x'' = -gamma x' + sqrt(2 kT gamma) W`, or without it.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'brownian_motion'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
