"""
# Next-generation reservoir computer

A next-generation reservoir computer has no random weights and no hidden units. Each variable keeps its last value,
and a ridge readout of the variables, their lags, and the products of those gives the next increment. Fitted to 10 time
units of the Lorenz flow, the readout then runs alone and follows the butterfly of the attractor.
"""

# %%
# The factory fits `reservoir(series, kind='next_generation', delay=2, order=2)` to 400 samples of the Lorenz flow, one
# every 0.025, with the ridge 2.5e-6 on features that are not standardized, as Gauthier et al. (2021) do. It writes the
# readout as one symbolic map per variable. One step of the system stands for one sample. The primary view shows the
# three variables of the forecast.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.neural_networks import LorenzFlowElement, NextGenerationReservoirSystem

SYSTEM = 'next_generation_reservoir'
STEP = 0.025
LYAPUNOV_TIME = 1 / 0.9056
VARIABLES = ['x', 'y', 'z']

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/reservoir', name='forecast', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
system = NextGenerationReservoirSystem()
forecast = simulate(system, observers=[observer])
for name, color in zip(VARIABLES, ((31, 119, 180), (255, 127, 14), (44, 160, 44)), strict=True):
    observer.recording_.log(f'reservoir/{name}', rr.SeriesLines(colors=[color], names=name), static=True)
    observer.recording_.send_columns(
        f'reservoir/{name}',
        indexes=[rr.TimeColumn('sim_time', duration=forecast.times)],
        columns=rr.Scalars.columns(scalars=forecast.values(f'reservoir.{name}')),
    )
observer.recording_.flush()


# %%
# Forecast against the true flow
# ------------------------------
# We train on a window of a long run of the flow and compare the forecast with the rest of the run. The two should agree
# within 10% of the attractor size for at least one Lyapunov time, then part as chaos amplifies the small error. After
# that the forecast should stay on the attractor, which the right panel compares with the true flow.


def lorenz_samples(samples):
    """Return samples of the Lorenz flow on its attractor."""
    rates = {
        'x': lambda s: 10.0 * (s.target.y - s.target.x),
        'y': lambda s: s.target.x * (28.0 - s.target.z) - s.target.y,
        'z': lambda s: s.target.x * s.target.y - 8.0 / 3.0 * s.target.z,
    }
    flow = System(
        elements=[LorenzFlowElement('flow')],
        interactions=[RateInteraction(name, target='flow', state=name, rate=rate) for name, rate in rates.items()],
    )
    solver = ScipyODESolver(method='RK23', rtol=1e-3, atol=1e-6)
    run = simulate(flow, until=5.0 + STEP * (samples - 1), solver=solver, dt=STEP)
    return np.column_stack([run.values(f'flow.{name}') for name in VARIABLES])[-samples:]


TRAINED = 402
HORIZON = 800
values = lorenz_samples(TRAINED + HORIZON + 1)
trained = NextGenerationReservoirSystem(series=values[: TRAINED + 1])
run = simulate(trained, until=HORIZON, dt=1)
predicted = np.column_stack([run.values(f'reservoir.{name}') for name in VARIABLES])[1:]
truth = values[TRAINED + 1 :]
size = np.ptp(values, axis=0).max()
errors = np.linalg.norm(predicted - truth, axis=1) / size
lyapunov_times = STEP * np.arange(1, HORIZON + 1) / LYAPUNOV_TIME
good = lyapunov_times[np.argmax(errors > 0.1)]

fig, (distance, attractor) = plt.subplots(1, 2, figsize=(10, 4))
distance.semilogy(lyapunov_times, errors, label='forecast error')
distance.axhline(0.1, color='0.6', linestyle='--', label='10% of the attractor size')
distance.set_xlabel('Lyapunov times')
distance.set_ylabel('error / attractor size')
distance.set_title(f'The forecast holds for {good:.1f} Lyapunov times')
distance.legend()
attractor.plot(truth[:, 0], truth[:, 2], color='0.7', linewidth=0.6, label='true flow')
attractor.plot(predicted[:, 0], predicted[:, 2], linewidth=0.6, label='forecast')
attractor.set_xlabel('x')
attractor.set_ylabel('z')
attractor.set_title('The forecast stays on the butterfly')
attractor.legend()
fig.tight_layout()
plt.show()
