"""
# Agent-based Lotka-Volterra

Prey and predators that walk on a grid, whose numbers cycle like the Lotka-Volterra equations.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'agent_based_lotka_volterra'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
