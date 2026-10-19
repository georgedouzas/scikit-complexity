"""
# Duffing

The forced Duffing oscillator `x'' = -delta x' - alpha x - beta x^3 + gamma cos(omega t)`, chaotic here.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'duffing'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=16)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
