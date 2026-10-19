"""
# Cusp normal forms

The cusp normal form `x' = β1 + β2 x - x³`, bistable inside the cusp `27 β1² < 4 β2³`.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'cusp_form'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
