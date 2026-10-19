"""
# Integrate-and-fire

The leaky integrate-and-fire neuron, `tau v' = -(v - rest) + R I`, which spikes and resets at a threshold.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'integrate_and_fire'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
