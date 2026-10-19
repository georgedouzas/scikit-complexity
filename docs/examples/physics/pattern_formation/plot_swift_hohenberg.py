"""
# Swift-Hohenberg

The Swift-Hohenberg equation `u_t = r u - (1 + laplacian)^2 u - u^3`, which forms rolls of length 2 pi.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'swift_hohenberg'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
