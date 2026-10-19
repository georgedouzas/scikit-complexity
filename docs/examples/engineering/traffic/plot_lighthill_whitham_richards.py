"""
# Lighthill-Whitham-Richards

The Lighthill-Whitham-Richards model, the conservation of cars with the Greenshields speed law.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'lighthill_whitham_richards'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
