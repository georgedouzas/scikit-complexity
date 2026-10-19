"""
# Logistic growth with a varying capacity

Logistic growth under a seasonal carrying capacity, which settles on a cycle of the season's period.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'varying_capacity_logistic'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
