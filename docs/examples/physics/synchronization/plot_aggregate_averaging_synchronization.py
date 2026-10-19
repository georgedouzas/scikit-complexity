"""
# Aggregate-averaging synchronization

Two logistic maps coupled by aggregate averaging, which synchronize their chaos for strong coupling.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'aggregate_averaging_synchronization'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
