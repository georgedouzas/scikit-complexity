"""
# Multiplicative noise conventions

Free particles in a box with position-dependent noise, read in the Itô, Stratonovich, or anti-Itô sense.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'multiplicative_noise'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=8)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
