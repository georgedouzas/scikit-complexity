"""
# Chemotherapy and oncolytic virotherapy

The uninfected tumour cells grow while the drug holds at its infusion balance, until the virus takes off and the
infected cells rise behind it.
"""

# %%
# The model is created by name from the catalog with the initial conditions of Malinzi: one unit of uninfected cells,
# a million cells, a tenth of a unit of virus, and a constant drug infusion. One unit of time is two days. The observer
# saves every frame with the default blueprint, whose primary view is the time series of the cells, the virus, and the
# drug.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'chemovirotherapy'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.2, observers=[observer]).run(until=32)

# %%
# Each therapy alone and both together
# ------------------------------------
# The same tumour is run with the drug alone, with the virus alone, and with both. Up to the end of the run, before the
# virus bursts through the tumour, the combination should leave the smallest tumour, about a tenth below either therapy
# alone.

arms = {'chemotherapy alone': create(MODEL, V=0.0), 'virus alone': create(MODEL, u1=0.0)}
fig, (curves, bars) = plt.subplots(1, 2, figsize=(10, 4))
finals = {}
for name, arm in arms.items():
    trajectory = Simulation(arm, dt=0.2).run(until=32).trajectory_
    total = trajectory['tumour.AU'] + trajectory['tumour.AV']
    curves.semilogy(2 * trajectory.times, total, label=name)
    finals[name] = total[-1]
combined = simulation.trajectory_
total = combined['tumour.AU'] + combined['tumour.AV']
curves.semilogy(2 * combined.times, total, label='combination', color='black')
finals['combination'] = total[-1]
curves.set_xlabel('time (days)')
curves.set_ylabel('tumour cells (millions)')
curves.set_title('The tumour under each therapy')
curves.legend()
bars.bar(list(finals), list(finals.values()), color=['tab:blue', 'tab:orange', 'black'])
bars.set_ylabel('tumour cells at day 64 (millions)')
bars.set_title('The combination leaves the smallest tumour')
fig.tight_layout()
plt.show()
