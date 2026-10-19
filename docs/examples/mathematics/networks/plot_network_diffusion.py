"""
# Diffusion

Diffusion on a network, `du_i/dt = D Σ_j A_ij (u_j - u_i)`, which spreads a unit mass to every node.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'network_diffusion'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
