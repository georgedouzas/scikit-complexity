"""
# Immune competition

Tumour cells and immune cells are distributions over activity classes, the progression of the tumour cells and the
activation of the immune cells. Tumour cells proliferate, active immune cells destroy them, and tumour cells inhibit
the immune cells. When the immune activation is above its threshold the tumour density decays to zero, and below it
the tumour escapes and grows.
"""

# %%
# The immune cells start at an activation of 0.75, above the threshold `growth / (kill immune) = 0.3`. The blueprint
# shows the tumour density, which decays to zero within a few time units.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'immune_competition'

# The distributions have no spatial view that shows the outcome, so the primary view is the tumour density over the run.
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/tumour_density', name='tumour density', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL)
simulation = Simulation(system, observers=[observer], dt=0.1, observables=describe(MODEL).observables_of(system)).run(
    until=10,
)

# %%
# The activation threshold
# ------------------------
# While `kill activation immune` exceeds `growth`, the immune cells destroy tumour cells faster than they divide. The
# tumour density after 20 time units should fall below its initial value above the activation 0.3, and rise above it
# below, where the growing tumour inhibits the immune cells further.


def final_density(activation):
    """Return the tumour density after 20 time units for an initial immune activation."""
    model_system = create(MODEL, activation=activation)
    run = Simulation(model_system, dt=20.0, observables=describe(MODEL).observables_of(model_system)).run(until=20)
    return run.trajectory_['tumour_density'][-1]


activations = np.linspace(0.0, 1.0, 5)
densities = [final_density(activation) for activation in activations]
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(activations, np.maximum(densities, 1e-12), marker='o', label='tumour density at t = 20')
ax.axhline(0.05, color='grey', linestyle=':', label='initial tumour density')
ax.axvline(0.3, color='black', linestyle='--', label='threshold growth / (kill immune)')
ax.set_xlabel('initial immune activation')
ax.set_ylabel('tumour density')
ax.set_title('The immune response clears the tumour above the threshold')
ax.legend()
plt.show()
