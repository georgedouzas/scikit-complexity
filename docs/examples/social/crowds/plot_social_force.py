"""
# Social force pedestrians

Two crowds walking in opposite directions along a periodic corridor, which form lanes.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'social_force'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
