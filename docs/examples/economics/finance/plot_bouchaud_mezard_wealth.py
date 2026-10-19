"""
# Bouchaud-Mézard wealth

Bouchaud and Mezard's wealth dynamics, random multiplicative returns with exchange between neighbours.

This example is scaffolded from the catalog and awaits curation.
"""

from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'bouchaud_mezard_wealth'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})
