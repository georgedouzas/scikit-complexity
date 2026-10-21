"""
# Adaptrode

A synapse with three weights on three time scales remembers its input at three depths. While a train of pulses
arrives, the fast weight follows every pulse, the medium weight builds up, and the slow weight builds up behind it.
After the train stops, the fast weight falls back to the medium one, and the slow weight keeps rising for a while and
holds the longest memory.
"""

# %%
# The model is created by name from the catalog with a pulse every ten steps until step 150. The observer saves every
# second step with the default blueprint, whose primary view is the time series of the three weights.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'adaptrode'

system = create(MODEL, period=10.0, stop=150.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=400)

# %%
# Three memories
# --------------
# Normalizing each weight by its value when the input stops, the fast weight should drop fastest, the medium weight
# more slowly, and the slow weight should first keep growing on what the faster weights hold and then decay slowest.

trajectory = simulation.trajectory_
stop = trajectory.times.searchsorted(150)
fig, (weights, decay) = plt.subplots(1, 2, figsize=(11, 4))
for name, label in [('w0', 'fast'), ('w1', 'medium'), ('w2', 'slow')]:
    weights.plot(trajectory.times, trajectory[f'synapse.{name}'], label=f'{label} {name}')
    decay.semilogy(
        trajectory.times[stop:] - 150,
        trajectory[f'synapse.{name}'][stop:] / trajectory[f'synapse.{name}'][stop],
        label=label,
    )
weights.axvline(150, color='grey', linestyle=':', label='input stops')
weights.set_xlabel('step')
weights.set_ylabel('weight')
weights.set_title('Learning while the pulses arrive')
weights.legend()
decay.set_xlabel('steps after the input stops')
decay.set_ylabel('weight / weight at the stop')
decay.set_title('Forgetting on three time scales')
decay.axhline(1.0, color='grey', linestyle=':')
decay.legend()
fig.tight_layout()
plt.show()
