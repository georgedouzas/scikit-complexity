"""
# Lorenz-96

Lorenz-96, `dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F` on a ring of sites around a latitude circle.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'lorenz_96'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
