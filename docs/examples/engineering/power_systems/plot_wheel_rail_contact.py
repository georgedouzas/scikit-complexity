"""
# Wheel and rail contact

A rail segment on a sleeper, loaded by the wheels of passing bogies.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'wheel_rail_contact'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=5)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
