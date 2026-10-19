"""
# Harvesting farmers

A socio-ecological model of farmer agents harvesting a logistically regrowing resource.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'harvesting'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=20)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
