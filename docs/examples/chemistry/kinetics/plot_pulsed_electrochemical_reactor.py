"""
# Pulsed electrochemical reactor

A stirred electrochemical reactor whose electrode is driven by a train of current pulses.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'pulsed_electrochemical_reactor'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=3)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
