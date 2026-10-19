"""
# Tent

The tent map `x' = μ min(x, 1 - x)`, chaotic with Lyapunov exponent `ln μ` for μ above 1.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'tent_map'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
