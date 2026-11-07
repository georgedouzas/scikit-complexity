"""
# Flux balance analysis

The steady-state balances of the metabolites and the bounds of the reactions confine the fluxes of a metabolic network,
and growth is the flux that an optimum picks among them. Raman's toy network of Table 8.2 grows on one substrate through
two parallel routes, so the optimum fixes the growth and leaves the split between the routes open.
"""

# %%
# The system is built from its class. The system holds one program under static time, so the run has one frame, and
# the primary view is the bar chart of the fluxes.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import sweep
from skcomplex.systems.biology.common.metabolism import flux_variability
from skcomplex.systems.biology.systems_biology import FluxBalanceSystem

SYSTEM = 'flux_balance'

system = FluxBalanceSystem().prepare().system_
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(system, observers=[observer], until=0.0)

# %%
# The ranges of the fluxes at the optimum
# ---------------------------------------
# Flux variability should fix `v4` to `v7`, `b1`, `b2`, and growth at the optimum, and leave `v1`, `v2`, and `v3` free
# over the range from 0 to 600, the two routes from `A` to `B`. The parsimonious solution should sit inside every range.

ranges = flux_variability(system).ranges
parsimonious = simulate(FluxBalanceSystem(parsimonious=True))
fluxes = [float(parsimonious.values(f'cell.v_{name}')[0]) for name in ranges.index]
fig, ax = plt.subplots(figsize=(9, 4))
positions = np.arange(len(ranges))
ax.bar(positions, ranges['maximum'] - ranges['minimum'], bottom=ranges['minimum'], width=0.5, label='flux variability')
ax.plot(positions, ranges['minimum'], '_', color='tab:blue', markersize=18)
ax.plot(positions, fluxes, 'o', color='tab:orange', label='parsimonious flux')
ax.set_xticks(positions, ranges.index)
ax.set_ylabel('flux (mmol/gDW/h)')
ax.legend(loc='upper left')
growth = float(trajectory.values('cell.v_vbio')[0])
ax.set_title(f'Growth {growth:.0f}: the routes v1, v2, and v3 range over 0 to 600, and the other fluxes are fixed')
fig.tight_layout()
plt.show()

# %%
# Growth against oxygen
# ---------------------
# On the E. coli core system, which needs the extra `metabolism`, a `sweep` over a grid of the oxygen
# uptake bound should give growth 0.2117 h⁻¹ without oxygen, a rise with oxygen, and 0.8739 h⁻¹ once oxygen no longer
# limits.

oxygen = np.linspace(0.0, 25.0, 11)
result = sweep(
    FluxBalanceSystem, {'oxygen': oxygen.tolist(), 'network': ['ecoli']}, until=0.0, outputs=['cell.objective_0'],
)
rows = result.rows
growth_rates = [float(np.ravel(row['cell.objective_0'])[-1]) for row in rows]
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(oxygen, growth_rates, 'o-')
ax.set_xlabel('largest oxygen uptake (mmol/gDW/h)')
ax.set_ylabel('growth (1/h)')
ax.set_title(
    f'Growth rises from {growth_rates[0]:.4f} to {growth_rates[-1]:.4f} 1/h, the plateau above an uptake of 21.8',
)
fig.tight_layout()
plt.show()
