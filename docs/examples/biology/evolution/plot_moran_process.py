"""
# Moran process

In a population of constant size, one individual reproduces and one dies at every step. A single mutant with a
selective advantage usually disappears by drift, but when it survives the first steps it sweeps through the
population and fixes, with the probability Moran computed.
"""

# %%
# Fifty individuals start with ten mutants that have a 20% advantage. The default blueprint shows the share of
# mutants, the mean of the trait, rising to fixation, with the spread of the trait peaking halfway.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'moran_process'

system = create(MODEL, size=50, mutants=10, selection=0.2)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=40)

# %%
# Fixation probability
# --------------------
# A single mutant of relative fitness `r` fixes with probability `(1 - 1 / r) / (1 - 1 / r^N)`. Repeating the
# process from one mutant many times, the share of runs where it fixes should match that formula, well above the
# neutral `1 / N`.


def mutants_of(trajectory):
    """Return the number of mutants at every frame."""
    frames = range(len(trajectory.times))
    return np.array([np.sum(trajectory.agents('population', frame).values['trait'] > 0) for frame in frames])


rng = np.random.default_rng(1)
size, selection, runs = 20, 0.5, 60
fixed = 0
for _ in range(runs):
    trajectory = Simulation(create(MODEL, size=size, selection=selection), dt=1, random_state=rng).run(until=150)
    fixed += mutants_of(trajectory.trajectory_)[-1] == size
r = 1 + selection
fig, (sweep, probability) = plt.subplots(1, 2, figsize=(11, 4))
sweep.plot(simulation.trajectory_.times, mutants_of(simulation.trajectory_))
sweep.set_xlabel('step')
sweep.set_ylabel('mutants')
sweep.set_title('Ten mutants with a 20% advantage sweep')
probability.bar(['simulated', 'Moran', 'neutral 1/N'], [fixed / runs, (1 - 1 / r) / (1 - r**-size), 1 / size])
probability.set_ylabel('fixation probability of one mutant')
probability.set_title(f'{runs} runs, N = {size}, r = {r}')
fig.tight_layout()
plt.show()
