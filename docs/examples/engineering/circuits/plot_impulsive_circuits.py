"""
# Circuits under periodic impulses

An RLC circuit driven by a periodic train of current impulses, `L C x'' + R C x' + x = A Σ δ(t - k T)`.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'impulsive_circuits'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
