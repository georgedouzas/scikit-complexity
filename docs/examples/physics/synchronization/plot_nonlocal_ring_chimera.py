"""
# Nonlocal ring chimera

Identical phase oscillators on a ring, each coupled to its `R` nearest neighbours on both sides.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'nonlocal_ring_chimera'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
