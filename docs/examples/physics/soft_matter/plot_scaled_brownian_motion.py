"""
# Scaled Brownian motion

Scaled Brownian motion, `dx = sqrt(2 K alpha (t + t0)^(alpha - 1)) dW`, with `<x^2> ~ 2 K t^alpha`.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'scaled_brownian_motion'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
