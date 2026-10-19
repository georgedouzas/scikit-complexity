"""
# Burridge-Knopoff spring-block fault

The Burridge-Knopoff fault, a chain of blocks pulled by a moving plate that sticks and slips.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'burridge_knopoff'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
