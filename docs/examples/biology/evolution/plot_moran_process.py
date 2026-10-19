"""
# Moran process

The Moran process, a population of constant size where fitter traits fix by selection and drift.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'moran_process'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
