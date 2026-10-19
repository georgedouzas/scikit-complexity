"""
# Motility-induced phase separation

Repulsive active Brownian disks of unit diameter, which phase separate into clusters at high speed.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'motility_induced_phase_separation'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
