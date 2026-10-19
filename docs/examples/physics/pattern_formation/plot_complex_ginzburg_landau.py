"""
# Complex Ginzburg-Landau

The complex Ginzburg-Landau equation `A_t = A + (1 + i b) laplacian(A) - (1 + i c) |A|^2 A`.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'complex_ginzburg_landau'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
