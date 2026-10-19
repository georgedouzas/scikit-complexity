"""
# Heterogeneous mean-field SIS

The SIS epidemic on a network with a power-law degree distribution, in heterogeneous mean field.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'heterogeneous_mean_field_sis'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
