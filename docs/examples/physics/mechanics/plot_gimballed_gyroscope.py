"""
# Gyroscope in a gimbal

A gyroscope in a gimbal: a rotor in an inner ring that turns in an outer ring about a fixed axis.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'gimballed_gyroscope'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
