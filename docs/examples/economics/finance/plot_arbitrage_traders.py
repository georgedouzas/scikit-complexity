"""
# Random-walk traders with arbitrage

Schwichtenberg's diffusion of a good by random-walk traders, minimally coupled to exchange rates.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'arbitrage_traders'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
