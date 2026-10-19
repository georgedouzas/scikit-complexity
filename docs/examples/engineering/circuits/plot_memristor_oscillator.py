"""
# Memristor oscillator

The simplest memristor oscillator: a capacitor, an inductor, and a memristor in series.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'memristor_oscillator'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=8)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
