"""
# Unbounded-alphabet automaton

An additive automaton over the nonnegative integers, `x'(n) = x(n - 1) + x(n)`, from a single 1.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'unbounded_alphabet_automaton'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
