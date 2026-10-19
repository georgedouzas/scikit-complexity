"""
# Diffusion automaton

A cellular automaton of diffusion, `c' = c + r Σ (c_neighbour - c)`, started from a square of ones.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'diffusion_automaton'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
