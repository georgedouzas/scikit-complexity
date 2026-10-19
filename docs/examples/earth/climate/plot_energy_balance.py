"""
# Zero-dimensional energy balance with ice albedo

The zero-dimensional energy balance of the Earth, bistable through the ice-albedo feedback.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'energy_balance'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
