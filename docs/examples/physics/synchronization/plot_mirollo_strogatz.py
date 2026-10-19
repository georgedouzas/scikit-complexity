"""
# Mirollo-Strogatz pulse-coupled oscillators

Integrate-and-fire oscillators coupled by pulses, which end up firing in unison.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'mirollo_strogatz'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
