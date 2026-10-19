"""
# Rosenzweig-MacArthur and rate tipping

Fast prey and slow predators whose carrying capacity shrinks at a rate, which tips them when it is fast.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'rosenzweig_macarthur'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
