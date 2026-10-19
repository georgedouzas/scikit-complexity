"""
# Takens-Bogdanov

The Takens-Bogdanov normal form `x' = y`, `y' = β1 + β2 x + x² + s x y`, where a saddle-node meets a Hopf.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'takens_bogdanov_form'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
