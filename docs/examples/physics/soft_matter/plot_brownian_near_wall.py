"""
# Brownian particle near a wall

A Brownian particle whose diffusion slows near a wall, `D(z) = D0 z² / (z² + a²)`, as it settles on it.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'brownian_near_wall'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
