"""
# Rössler

The Rössler model, a chaotic flow with one folded band for its standard parameters.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'rossler'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=16)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
