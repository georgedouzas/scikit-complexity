"""
# Diffusion in a speckle field

Brownian particles in a random speckle field, trapped at hotspots for longer as the field grows stronger.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'diffusion_in_a_speckle_field'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
