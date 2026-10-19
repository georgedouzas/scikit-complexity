"""
# Swarmtaxis

The swarmtaxis algorithm, which moves a swarm of robots towards a beacon that only some of them see.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'swarmtaxis'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
