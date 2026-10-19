"""
# Pheromone foraging

Ants that find food and lay a pheromone trail home, which other ants follow: stigmergy.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'pheromone_foraging'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=3)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
