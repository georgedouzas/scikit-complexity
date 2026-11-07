"""
# Hora and Tempus

Simon's fable, told by Meadows: two watchmakers build watches of 1000 parts and are interrupted by customers' calls.
Tempus assembles a watch in one go, so an interruption makes it fall to pieces. Hora builds stable subassemblies of
ten elements, so an interruption loses only the assembly in hand. Hierarchy wins by orders of magnitude.
"""

# %%
# The probability of an interruption is 1% per action, and the run lasts 20,000 actions. The default blueprint shows
# the progress and the finished watches of both watchmakers.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'watchmakers'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=100)
simulation = Simulation(create_system(SYSTEM), observers=[observer], random_state=0).prepare().run(until=20000)

# %%
# Watches finished
# ----------------
# A task of `k` actions interrupted with the probability `p` per action takes `((1 - p)^-k - 1) / p` actions on
# average, so Hora should finish a watch every 1173 actions, while Tempus needs 2.3 million and finishes none.

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
watches = np.array([trajectory.agents('watchmakers', frame).values['watches'] for frame in range(len(times))])
progress = np.array([trajectory.agents('watchmakers', frame).values['progress'] for frame in range(len(times))])
fig, (finished, work) = plt.subplots(1, 2, figsize=(10, 4))
finished.plot(times, watches[:, 0], label='Hora')
finished.plot(times, watches[:, 1], label='Tempus')
finished.plot(times, times / (111 * (0.99**-10 - 1) / 0.01), ':', color='black', label='expected for Hora')
finished.set_xlabel('actions')
finished.set_ylabel('finished watches')
finished.set_title('Stable subassemblies pay')
finished.legend()
work.plot(times[:3000], progress[:3000, 1], label='Tempus, parts in hand')
work.set_xlabel('actions')
work.set_ylabel('parts')
work.set_title('Tempus starts over at every call')
work.legend()
fig.tight_layout()
plt.show()
