"""
# Tunnel-diode oscillator

A tunnel-diode oscillator: an inductor and a capacitor in parallel with a negative-resistance diode.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'tunnel_diode_oscillator'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
