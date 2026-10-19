"""
# Izhikevich

The Izhikevich neuron, whose four parameters give regular spiking, bursting, chattering, and more.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'izhikevich'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=8)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
