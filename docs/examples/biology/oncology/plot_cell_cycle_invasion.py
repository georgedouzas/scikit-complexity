"""
# Cell cycle and invasion of a scratch

Melanoma cells on a triangular lattice pass through the phases G1, early S, and S/G2/M of the fluorescent cell cycle
indicator, divide next to vacancies, and move into vacant sites. Cells at the edges of a scratch through a monolayer
cycle and move into it, while the crowded cells behind them stay in G1.
"""

# %%
# The lattice holds 12 rows of 24 sites, full outside a vacant scratch of the middle quarter of the columns, with the
# rates of Carr et al. per hour. The run lasts 48 hours, as the experiments do. The observer saves every frame with the
# default blueprint.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import calibrate
from skcomplex.systems.biology.oncology import CellCycleInvasionSystem

SYSTEM = 'cell_cycle_invasion'
PHASES = ('g1', 'es', 'sg2m')

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(CellCycleInvasionSystem(), observers=[observer], random_state=0)

# %%
# Cells per phase and the closing scratch
# ---------------------------------------
# The cells per phase are the observables of the system. The cells inside the scratch should rise as the edges invade
# it, and the cells in G1 should hold most of the monolayer, since a cell leaves G1 only next to a vacancy.

columns = np.arange(24 * 12) % 24
inside = (columns >= 9) & (columns < 15)
occupied = sum(np.asarray(trajectory.get_values(f'cells.{phase}')) for phase in PHASES)
fig, (phases, scratch) = plt.subplots(1, 2, figsize=(10, 4))
for phase, colour in zip(PHASES, ('tab:red', 'gold', 'tab:green'), strict=True):
    phases.plot(trajectory.times, trajectory.get_values(f'{phase}_cells'), color=colour, label=phase)
phases.set_xlabel('time (hours)')
phases.set_ylabel('cells')
phases.set_title('Cells per phase')
phases.legend()
scratch.plot(trajectory.times, occupied[:, inside].sum(axis=1), color='black')
scratch.set_xlabel('time (hours)')
scratch.set_ylabel('cells inside the scratch')
scratch.set_title('The edges invade the scratch')
fig.tight_layout()
plt.show()

# %%
# Calibrating the transition out of G1 by ABC-SMC
# -----------------------------------------------
# The counts of the three phases at 48 hours are the summaries of Carr et al. ABC-SMC compares them for every
# simulation with those of one run at `Rr = 0.1`, with a uniform prior on `Rr` between 0.01 and 0.5 and a small budget
# of simulations. The weighted posterior should concentrate around the true rate.

truth = simulate(CellCycleInvasionSystem(12, 6, Rr=0.1), until=48, random_state=1)
times = np.arange(0.0, 49.0, 12.0)
data = {
    f'{phase}_cells': (times, np.interp(times, truth.times, truth.get_values(f'{phase}_cells'))) for phase in PHASES
}


def final_counts(series):
    """Return the cells of every phase at the last data time."""
    return [series[f'{phase}_cells'][-1] for phase in PHASES]


posterior = calibrate(
    CellCycleInvasionSystem(12, 6),
    data,
    {'Rr': (0.01, 0.5)},
    method='abc_smc',
    summary=final_counts,
    n_samples=60,
    budget=400,
    random_state=0,
)
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(posterior.samples['Rr'], bins=20, weights=posterior.weights, color='tab:blue', alpha=0.7, label='posterior')
ax.axvline(0.1, color='black', linestyle='--', label='true rate')
ax.axvline(posterior.estimates['Rr'], color='tab:red', label=f'median {posterior.estimates["Rr"]:.3f}')
ax.set_xlabel('rate from G1 to early S (per hour)')
ax.set_ylabel('weight')
ax.set_title('ABC-SMC on the counts of the phases')
ax.legend()
plt.show()
