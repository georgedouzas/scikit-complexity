"""
# Two-mass spring-damper between walls

Two masses between two walls, tied to the walls and to each other by springs, with dampers and forces.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'two_mass_spring_damper'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=3)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
