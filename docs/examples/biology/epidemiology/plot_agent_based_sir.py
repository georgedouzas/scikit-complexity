"""
# Agent-based SIR with mortality, waning immunity, incubation, and lockdown

People who walk on a grid and infect those in their cell, with optional extensions of the SIR model.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'agent_based_sir'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0}, random_state=0)
