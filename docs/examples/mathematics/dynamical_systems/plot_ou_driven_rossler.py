"""
# Ornstein-Uhlenbeck driven Rössler

A Rössler flow whose control parameter `eta` wanders slowly as an Ornstein-Uhlenbeck process. As the control drifts,
the flow passes through different regimes of its bifurcation diagram, so the trajectory does not settle on one
attractor but smears a band of loops of varying size. The view shows this smeared band in three dimensions, with the
spikes of `z` that lift the trajectory out of the plane.
"""

# %%
# The model is created by name from the catalog with the parameters `a = 0.2`, `b = 0.4`, and `c = 5.743` of Cestnik
# and Martens, the correlation time `tau = 50`, and the noise amplitude `rho = 1.5`, so the control has the standard
# deviation `rho / sqrt(2 tau) = 0.15`. The run of 600 time units uses the Numba backend. After the run, the trajectory
# is logged once as a line in three dimensions, and the blueprint puts that view first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.signal import find_peaks

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'ou_driven_rossler'

solver = describe(MODEL).defaults['solver']
# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
camera = rrb.EyeControls3D(position=(0.0, -40.0, 18.0), look_target=(1.0, 0.0, 6.0), eye_up=(0.0, 0.0, 1.0))
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin='/attractor', name='attractor', eye_controls=camera, line_grid=False),
        rrb.TimeSeriesView(origin='/rossler', name='x, y, z, eta', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=25)
simulation = Simulation(
    create(MODEL, rho=1.5),
    dt=0.1,
    solver=solver,
    observers=[observer],
    random_state=0,
    backend='numba',
).run(until=600)
points = np.column_stack([simulation.trajectory_[f'rossler.{state}'] for state in 'xyz'])[500:]
observer.recording_.log('attractor', rr.LineStrips3D([points], radii=0.05, colors=[255, 170, 40]), static=True)
observer.recording_.flush()

# %%
# A bifurcation diagram from one noisy run
# ----------------------------------------
# Every local maximum of `x`, plotted against the control at that moment, samples the bifurcation diagram of the flow
# at that control. Over a long run, the maxima of the noisy flow should fall near the maxima of the flow held at fixed
# controls, the diagram that Cestnik and Martens reconstruct from such data. The control should be Gaussian with the
# standard deviation 0.15.

long = Simulation(create(MODEL, rho=1.5), dt=0.1, solver=solver, random_state=1, backend='numba').run(until=3000)
x, eta = long.trajectory_['rossler.x'], long.trajectory_['rossler.eta']
peaks = find_peaks(x)[0]
fig, (diagram, density) = plt.subplots(1, 2, figsize=(10, 4))
diagram.plot(eta[peaks], x[peaks], '.', markersize=3, alpha=0.5, label='noisy control')
for control in np.linspace(-0.3, 0.3, 13):
    fixed = create(MODEL, eta=control, tau=1e9, rho=0.0)
    held = Simulation(fixed, dt=0.1, solver=solver, backend='numba').run(until=800).trajectory_['rossler.x'][3000:]
    maxima = held[find_peaks(held)[0]]
    diagram.plot(np.full(maxima.size, control), maxima, 'k.', markersize=3)
diagram.plot([], [], 'k.', label='fixed control')
diagram.set_xlabel('control eta')
diagram.set_ylabel('local maxima of x')
diagram.set_title('Maxima trace the bifurcation diagram')
diagram.legend()
density.hist(eta, bins=60, density=True, alpha=0.6)
grid = np.linspace(-0.6, 0.6, 200)
density.plot(grid, np.exp(-(grid**2) / (2 * 0.15**2)) / np.sqrt(2 * np.pi * 0.15**2), 'k--', label='N(0, 0.15^2)')
density.set_xlabel('control eta')
density.set_ylabel('density')
density.set_title('The control is an Ornstein-Uhlenbeck process')
density.legend()
fig.tight_layout()
plt.show()
