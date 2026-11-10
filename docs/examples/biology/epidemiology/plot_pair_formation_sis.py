"""
# Pair formation SIS

People hold a fixed number of sexual partners and replace a partner as soon as a partnership ends. With the same
infection pressure and the same rate of new partners, concurrent partnerships keep more people infected than serial
monogamy, because the infection passes on before the partnership that brought it ends.
"""

# %%
# The example builds `PairFormationSISSystem` with three concurrent partners per person, run by the Gillespie solver.
# The observer saves every frame with the default blueprint, whose primary view is the network of current partnerships,
# with the infected people bright, as partnerships dissolve and form.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.epidemiology import PairFormationSISSystem

SYSTEM = 'pair_formation_sis'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(PairFormationSISSystem(n=200, k=3), observers=[observer], random_state=1, until=30)

# %%
# Prevalence and concurrency
# --------------------------
# At a fixed infection pressure `k beta` and a fixed rate of new partners `k sigma`, the prevalence should rise with
# the number `k` of concurrent partners.

fig, ax = plt.subplots(figsize=(7, 4.5))
for k in (1, 2, 4):
    run = simulate(PairFormationSISSystem(n=300, k=k), random_state=0, until=30)
    ax.plot(run.times, run.get_values('people.i').mean(axis=1), label=f'k = {k}')
ax.plot(trajectory.times, trajectory.get_values('people.i').mean(axis=1), ':', label='run above')
ax.set_xlabel('time')
ax.set_ylabel('prevalence')
ax.set_title('Concurrency raises the prevalence at a fixed contact rate')
ax.set_ylim(0, 1)
ax.legend()
plt.show()
