"""
# Intermittency maps

The Pomeau-Manneville map `x' = x + x^z (mod 1)`, whose long laminar phases near 0 end in chaotic bursts.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'intermittency_map'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
