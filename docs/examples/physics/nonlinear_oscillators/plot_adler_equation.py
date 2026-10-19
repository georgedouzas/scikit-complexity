"""
# Adler equation

The Adler equation `Δφ' = Δω + ε sin Δφ` for the phase difference of an oscillator and its drive.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'adler_equation'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
