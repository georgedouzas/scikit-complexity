"""
# Schrödinger equation as real and imaginary fields

A Gaussian wave packet that follows the Schrödinger equation, as its real and imaginary parts.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'schrodinger'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
