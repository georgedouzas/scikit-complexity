"""
# Cubic autocatalysis

Cubic autocatalysis `A + 2B -> 3B` with the decay `B -> C` in a continuously fed reactor.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'cubic_autocatalysis'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
