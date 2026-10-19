"""
# Replicator dynamics and hawks and doves

The replicator dynamics of a game, by default hawks and doves, which settle at a share `V / C` of hawks.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'replicator_dynamics'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
