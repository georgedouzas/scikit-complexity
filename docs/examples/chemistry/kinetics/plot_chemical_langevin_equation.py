"""
# Chemical Langevin equation of the Schlögl reaction

The chemical Langevin equation of Schlögl's bistable reaction, whose noise switches between two states.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'chemical_langevin_equation'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
