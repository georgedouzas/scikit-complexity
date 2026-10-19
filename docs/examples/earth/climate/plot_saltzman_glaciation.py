"""
# Saltzman-Maasch glaciation

The glacial cycles of Saltzman and Maasch, a self-sustained oscillation of ice, carbon, and ocean.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'saltzman_glaciation'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
