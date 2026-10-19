"""
# Standard

The Chirikov standard map `p' = p + K sin θ`, `θ' = θ + p'`, with global chaos above K ≈ 0.9716.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'standard_map'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
