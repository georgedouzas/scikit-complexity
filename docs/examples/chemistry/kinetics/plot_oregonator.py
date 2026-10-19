"""
# Oregonator

The scaled Oregonator, the reduction of the FKN mechanism of the Belousov-Zhabotinsky reaction.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'oregonator'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=3)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
