"""
# Ornstein-Uhlenbeck with weakening restoring force

Brownian particles held by a restoring force that weakens with distance.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'weakening_ornstein_uhlenbeck'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
