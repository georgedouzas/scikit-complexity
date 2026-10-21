"""
# Spatial prisoner's dilemma

Nowak and May's spatial prisoner's dilemma: players on a lattice play the weak dilemma with themselves and their
eight neighbours and then copy the strategy of the most successful player around them. Defection pays more in every
encounter, yet clusters of cooperators protect each other, and for intermediate temptations the two strategies
coexist in an ever-changing, chaotic patchwork with a stable share of cooperators.
"""

# %%
# On a 50 by 50 periodic lattice, 90 percent of the players start as cooperators and the temptation to defect is
# `b = 1.85`. The blueprint shows the lattice, cooperators in yellow and defectors in purple, after 100 rounds of
# imitation.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'spatial_prisoners_dilemma'
ROUNDS = 100

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
system = create(MODEL, temptation=1.85, random_state=0)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=ROUNDS)

# %%
# Three regimes of the temptation
# -------------------------------
# A cluster of cooperators grows when its boundary cooperators, which earn from several cooperators, beat the
# defectors next to them, and for the eight-neighbour lattice the outcome changes only at the temptations where these
# payoff comparisons switch. Below `b = 1.8` cooperators should win most of the lattice, between 1.8 and 2 the share of
# cooperators should fluctuate around the value of about 0.3 that Nowak and May found for the chaotic regime, and
# above 2 defectors should take over.


def cooperators(temptation, rounds=ROUNDS):
    """Return the share of cooperators after every round."""
    run = Simulation(create(MODEL, temptation=temptation, random_state=0), random_state=0).run(until=rounds)
    return run.trajectory_['players.strategy'].reshape(rounds + 1, -1).mean(axis=1)


temptations = np.linspace(1.05, 2.45, 15)
fig, (series, sweep) = plt.subplots(1, 2, figsize=(10, 4))
for index, temptation in enumerate((1.6, 1.85, 2.1)):
    shares = simulation.trajectory_['players.strategy'].reshape(ROUNDS + 1, -1).mean(axis=1)
    shares = shares if temptation == 1.85 else cooperators(temptation)
    series.plot(shares, color=f'C{index}', label=f'b = {temptation}')
series.axhline(12 * np.log(2) - 8, color='grey', linestyle=':', label='12 ln 2 - 8')
series.set_xlabel('round')
series.set_ylabel('share of cooperators')
series.set_title('Cooperation over time')
series.legend()
sweep.plot(temptations, [cooperators(temptation)[-50:].mean() for temptation in temptations], 'o-')
for edge in (1.8, 2.0):
    sweep.axvline(edge, color='grey', linestyle='--', lw=0.8)
sweep.set_xlabel('temptation b')
sweep.set_ylabel('share of cooperators, last 50 rounds')
sweep.set_title('Cooperation, chaos, and defection')
fig.tight_layout()
plt.show()
