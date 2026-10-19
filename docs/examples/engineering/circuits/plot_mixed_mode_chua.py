"""
# Modified Chua with mixed-mode oscillations

A modified Chua circuit on three time scales, whose oscillations mix small and large amplitudes.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'mixed_mode_chua'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
