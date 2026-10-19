"""
# ENSO delay oscillator

The delayed oscillator of El Niño of Suarez and Schopf, `dT/dt = T - T^3 - alpha T(t - delta)`.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'enso'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
