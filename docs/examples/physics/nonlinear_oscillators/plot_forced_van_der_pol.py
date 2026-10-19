"""
# Forced Van der Pol

The Van der Pol oscillator driven by a periodic force, `x'' = μ (1 - x²) x' - x + A cos(ω t)`.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'forced_van_der_pol'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=8)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
