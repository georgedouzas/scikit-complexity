"""
# Hénon-Heiles

The Hénon-Heiles model of a star in a galactic potential, whose motion is chaotic at higher energies.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'henon_heiles'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=16)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
