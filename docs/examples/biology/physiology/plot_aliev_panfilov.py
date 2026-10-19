"""
# Aliev-Panfilov cardiac tissue

The Aliev-Panfilov model of cardiac tissue, started from a broken wave that curls into a spiral.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'aliev_panfilov'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
