"""
# Chua double and multiple scrolls

Chua's circuit, whose double-scroll or multiscroll attractor is chaotic.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'chua'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=8)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
