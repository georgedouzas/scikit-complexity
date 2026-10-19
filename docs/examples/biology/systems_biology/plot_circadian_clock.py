"""
# Drosophila circadian clock

Goldbeter's model of the Drosophila circadian clock, PER that represses its own transcription.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'circadian_clock'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
