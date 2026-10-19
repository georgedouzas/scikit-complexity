"""
# Finite-time blow-up

Hyperbolic growth `x' = a x^p`, which diverges at the finite time `x0^(1 - p) / (a (p - 1))` for p > 1.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'finite_time_blow_up'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
