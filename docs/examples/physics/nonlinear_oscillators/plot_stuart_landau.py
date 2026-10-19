"""
# Stuart-Landau

The Stuart-Landau oscillator `z' = (μ + i ω) z - (1 + i β) |z|² z`, the normal form of a Hopf bifurcation.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'stuart_landau'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=3)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
