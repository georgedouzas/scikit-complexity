"""
# Rothman excitable carbon cycle

Rothman's excitable carbon cycle, where a large enough perturbation of the ocean sets off an excursion.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'carbon_cycle'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
