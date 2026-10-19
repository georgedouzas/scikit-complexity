"""
# FitzHugh-Nagumo ODE

The FitzHugh-Nagumo neuron, the two-variable reduction of Hodgkin and Huxley's excitable membrane.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'fitzhugh_nagumo_ode'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
