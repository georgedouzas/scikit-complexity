"""
# Koopman model predictive control

Three nonlinear plants follow a reference they would miss on their own. The controller knows no equation of a plant.
It lifts the states to monomials, predicts with a linear surrogate fitted to runs under a random input, and solves a
bounded least-squares problem at every sample. The primary view shows the forced Duffing oscillator tracking `-sin(t)`.
"""

# %%
# The factory fits `ModeDecompositionAnalysis(dictionary=PolynomialFeatures(3), inputs=['input.u'])` to 20 runs of 200
# samples each. The system then nests the controller under discrete time and the plant under continuous time, and
# `CoSimulationSolver` holds the input between samples. One time unit of a run is one sample, 0.1 time units of the
# Duffing oscillator.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create, describe, run

MODEL = 'koopman_model_predictive_control'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/tracking', name='output and reference', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
trajectory = (
    Simulation(system, dt=1.0, observers=[observer], observables=describe(MODEL).observables_of(system))
    .check()
    .prepare()
    .run(until=200.0)
    .trajectory_
)
time = [rr.TimeColumn('sim_time', duration=trajectory.times)]
for name, values, color in (
    ('output', trajectory.values('plant/state.y'), (31, 119, 180)),
    ('reference', trajectory.values('reference'), (214, 39, 40)),
):
    observer.recording_.log(f'tracking/{name}', rr.SeriesLines(colors=[color], names=name), static=True)
    observer.recording_.send_columns(f'tracking/{name}', indexes=time, columns=rr.Scalars.columns(scalars=values))
observer.recording_.flush()


# %%
# Controlled against uncontrolled
# -------------------------------
# For each plant the controlled output should stay closer to the reference than the output under no input. The surrogate
# predicts over the horizon of the review of Brunton et al. and misses what its monomials cannot express, so the
# comparison is against the uncontrolled plant and not against a bound on the error.


def tracked(plant, *, controlled):
    """Return the times, the output, the reference, and the mean squared error of a run of a plant."""
    until = 300.0 if plant == 'motor' else 150.0
    found = run(MODEL, plant=plant, controlled=controlled, settings={'until': until}).trajectory_
    output, reference = found.values('plant/state.y'), found.values('reference')
    return found.times, output, reference, np.mean((output - reference) ** 2)


fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, plant in zip(axes, ('duffing', 'van_der_pol', 'motor'), strict=True):
    times, output, reference, error = tracked(plant, controlled=True)
    _, free, _, free_error = tracked(plant, controlled=False)
    ax.plot(times, reference, color='0.5', linestyle='--', label='reference')
    ax.plot(times, output, label=f'controlled, MSE {error:.3g}')
    # The motor without input settles far below the range of the panel, so the panel reports only its error.
    label = f'no input, MSE {free_error:.3g}'
    if plant == 'motor':
        ax.plot([], [], color='tab:orange', linewidth=0.8, label=label)
    else:
        ax.plot(times, free, color='tab:orange', linewidth=0.8, label=label)
    ax.set_title(plant.replace('_', ' '))
    ax.set_xlabel('samples')
    ax.set_ylabel('output')
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
