"""
# Stochastic resonance

A particle in a double well, `dx = (x - x^3 + A cos(2 pi f t)) dt + sqrt(2 D) dW`.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'stochastic_resonance'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
