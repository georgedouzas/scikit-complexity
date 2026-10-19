"""
# Polyhomeostatic neuron

A neuron that adapts its gain and threshold so its firing rate approaches an exponential distribution.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'polyhomeostatic_neuron'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
