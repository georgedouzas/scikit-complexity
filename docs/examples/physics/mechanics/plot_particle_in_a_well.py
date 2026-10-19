"""
# Particle in the well `k x^2 e^-x`

A particle moving without friction in the potential `U(x) = k x² e^-x`.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'particle_in_a_well'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
