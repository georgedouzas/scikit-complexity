"""
# Insider sabotage

The insider sabotage model of employees whose risk of sabotage builds up over the days of a year.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'insider_sabotage'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
