"""
# Restricted three-body problem

A light body moving in the plane of two primaries on circular orbits, in the frame that rotates with them.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'restricted_three_body'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
