"""
# Cat map

Arnold's cat map `(x, y) → (2x + y, x + y) (mod 1)`, an area-preserving hyperbolic map of the torus.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'cat_map'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
