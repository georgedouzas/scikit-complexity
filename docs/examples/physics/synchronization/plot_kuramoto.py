"""
# Kuramoto

Kuramoto oscillators coupled all to all, `theta_i' = omega_i + (K / N) sum_j sin(theta_j - theta_i)`.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'kuramoto'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
