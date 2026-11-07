"""
# Chemotherapy and oncolytic virotherapy

The uninfected tumour cells grow while the drug holds at its infusion balance, until the virus takes off and the
infected cells rise behind it.
"""

# %%
# The system is built from its class with the initial conditions of Malinzi: one unit of uninfected cells, a million
# cells, a tenth of a unit of virus, and a constant drug infusion. One unit of time is two days. The observer saves
# every frame with the default blueprint, whose primary view is the time series of the cells, the virus, and the drug.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import ChemovirotherapySystem

SYSTEM = 'chemovirotherapy'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(ChemovirotherapySystem(), dt=0.2, observers=[observer])

# %%
# Each therapy alone and both together
# ------------------------------------
# The same tumour is run with the drug alone, with the virus alone, and with both. Up to the end of the run, before the
# virus bursts through the tumour, the combination should leave the smallest tumour, about a tenth below either therapy
# alone.

arms = {
    'chemotherapy alone': ChemovirotherapySystem(V=0.0).prepare().system_,
    'virus alone': ChemovirotherapySystem(u1=0.0).prepare().system_,
}
fig, (curves, bars) = plt.subplots(1, 2, figsize=(10, 4))
finals = {}
for name, arm in arms.items():
    trajectory = simulate(arm, dt=0.2, until=32)
    total = trajectory.values('tumour.AU') + trajectory.values('tumour.AV')
    curves.semilogy(2 * trajectory.times, total, label=name)
    finals[name] = total[-1]
combined = trajectory
total = combined.values('tumour.AU') + combined.values('tumour.AV')
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
