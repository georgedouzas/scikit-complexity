"""
# Prisoner's dilemma tournament

Agents play six-round prisoner's dilemmas against each other with strategies coded as genomes that answer the
opponent's last two moves. The fittest reproduce with rare mutations, and the population never settles: cooperative
eras, when reciprocating strategies dominate, alternate with eras of exploitation and mutual defection.
"""

# %%
# One hundred agents start from random genomes and evolve for 3000 steps. The default blueprint shows the mean score
# per round, which wanders between about 1.4, close to mutual defection at 1, and 2.7, close to mutual cooperation
# at 3.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.cooperation import PrisonersDilemmaTournamentSystem

SYSTEM = 'prisoners_dilemma_tournament'

system = PrisonersDilemmaTournamentSystem(random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=12)
trajectory = simulate(system, until=3000, observers=[observer], random_state=0)

# %%
# Eras of strategies
# ------------------
# Each gene answers one history of the opponent's last moves. The share of agents that cooperate on each history
# should switch between eras in step with the mean score: the score is high when the answer to the opponent's
# cooperation (genes `C` and `CC`) is cooperation, and falls when exploiters that defect after `C` spread.

frames = range(0, len(trajectory.times), 10)
genes = np.array([np.asarray(trajectory.get_agents('agents', frame).values['genome']).mean(axis=0) for frame in frames])
scores = [np.mean(trajectory.get_agents('agents', frame).values['fitness']) for frame in frames]
times = np.asarray(trajectory.times)[list(frames)]
fig, (strategy, score) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, height_ratios=[2, 1])
strategy.imshow(genes.T, aspect='auto', cmap='viridis', extent=(times[0], times[-1], 6.5, -0.5), vmin=0, vmax=1)
strategy.set_yticks(range(7), ['start', 'C', 'D', 'CC', 'CD', 'DC', 'DD'])
strategy.set_ylabel("opponent's last moves")
strategy.set_title('Share of agents that cooperate after each history (yellow: all cooperate)')
score.plot(times, scores)
score.set_xlabel('step')
score.set_ylabel('mean score per round')
fig.tight_layout()
plt.show()
