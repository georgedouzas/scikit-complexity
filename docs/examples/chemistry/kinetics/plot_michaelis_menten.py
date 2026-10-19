"""
# Michaelis-Menten as ODEs and as a jump process

The enzyme reaction `E + S <-> C -> E + P` of Michaelis and Menten, as rate equations or as jumps.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'michaelis_menten'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
