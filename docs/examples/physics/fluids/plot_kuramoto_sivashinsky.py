"""
# Kuramoto-Sivashinsky

The Kuramoto-Sivashinsky equation `u_t = -u u_x - u_xx - u_xxxx`, spatiotemporally chaotic.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'kuramoto_sivashinsky'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
