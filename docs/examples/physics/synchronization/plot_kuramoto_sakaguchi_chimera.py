"""
# Kuramoto-Sakaguchi chimera

Two populations of identical phase oscillators with phase lag, where one synchronizes and one drifts.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'kuramoto_sakaguchi_chimera'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
