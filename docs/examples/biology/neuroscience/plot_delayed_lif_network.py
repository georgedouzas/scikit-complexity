"""
# Networks of leaky integrate-and-fire neurons with delays

A random network of leaky integrate-and-fire neurons whose spikes reach their targets after a delay.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'delayed_lif_network'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
