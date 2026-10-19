"""
# Parametric pendulum

A pendulum whose suspension point oscillates vertically, `θ'' + c θ' + (g / l)(1 + ε cos Ω t) sin θ = 0`.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'parametric_pendulum'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=6)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
