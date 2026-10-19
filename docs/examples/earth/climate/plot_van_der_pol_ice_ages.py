"""
# Van der Pol ice ages

Glacial cycles as a Van der Pol relaxation oscillator, optionally paced by orbital forcing.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'van_der_pol_ice_ages'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
