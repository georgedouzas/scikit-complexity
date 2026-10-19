"""
# Ott-Antonsen reduction

The Ott-Antonsen equation for the order parameter of infinitely many Kuramoto oscillators.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'ott_antonsen'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
