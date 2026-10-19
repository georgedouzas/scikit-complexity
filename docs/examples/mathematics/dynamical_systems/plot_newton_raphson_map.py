"""
# Newton-Raphson map

Newton's method for `z³ = 1` in the complex plane, `z' = (2z³ + 1) / (3z²)`, in real coordinates.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'newton_raphson_map'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
