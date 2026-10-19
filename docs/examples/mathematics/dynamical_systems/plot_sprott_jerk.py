"""
# Sprott jerk

Sprott's jerk equation `x''' = -A x'' - x' + |x| - 1`, chaotic near A = 0.6.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'sprott_jerk'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=8)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
