"""
# Landau mean-field relaxation

The relaxation of a Landau order parameter, to `sqrt(a (Tc - T) / b)` below `Tc` and to 0 above it.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'landau_relaxation'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
