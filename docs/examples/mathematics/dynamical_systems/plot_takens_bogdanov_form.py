"""
# Takens-Bogdanov normal form

Near the Takens-Bogdanov point three bifurcations meet. At `beta2 = -1`, lowering `beta1` through 0 turns the stable
focus into a small limit cycle in a Hopf bifurcation; the cycle grows, slows down, and near `beta1 = -0.21` it
touches the saddle and disappears in a homoclinic bifurcation, after which orbits escape.
"""

# %%
# The run starts next to the focus at `beta1 = -0.1`, where the focus repels and the orbit spirals out onto the
# limit cycle. The model logs `x` and `y` as time series. After the run, its orbit is logged once as a line in the phase
# plane together with the separatrix of the saddle, and the blueprint puts this view first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.signal import find_peaks

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'takens_bogdanov_form'


def focus(beta1, beta2=-1.0):
    """Return the position of the equilibrium of the smaller root, the focus."""
    return (-beta2 - np.sqrt(beta2**2 - 4 * beta1)) / 2


blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/phase_plane',
            name='phase plane',
            visual_bounds=rrb.VisualBounds2D(x_range=[-100, 125], y_range=[-85, 85]),
        ),
        rrb.TimeSeriesView(origin='/form', name='x, y'),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, x=focus(-0.1) + 0.02, y=0.0, beta1=-0.1)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=8)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=200)
saddle = (1 + np.sqrt(1 + 0.4)) / 2
outside = Simulation(create(MODEL, x=saddle - 0.01, y=0.0, beta1=-0.1), dt=0.1).run(until=60).trajectory_
strips = [np.column_stack([100 * run['form.x'], -100 * run['form.y']]) for run in (simulation.trajectory_, outside)]
observer.recording_.log(
    'phase_plane',
    rr.LineStrips2D(strips, radii=0.6, colors=[[255, 170, 40], [90, 170, 255]]),
    static=True,
)
observer.recording_.log(
    'phase_plane/saddle',
    rr.Points2D([[100 * saddle, 0]], radii=2.5, colors=[230, 60, 60]),
    static=True,
)
observer.recording_.flush()

# %%
# From Hopf to homoclinic
# -----------------------
# Between the Hopf bifurcation at `beta1 = 0` and the homoclinic one, found here between -0.21 and -0.215 (the
# small-parameter estimate `-6 beta2² / 25` gives -0.24), the amplitude of the cycle
# should grow from 0 and its period should grow from `2π / √(-beta2) = 2π` without bound, logarithmically, as the
# cycle approaches the saddle and spends ever longer near it.

values = -np.linspace(0.005, 0.21, 21)
amplitudes, periods = [], []
for beta1 in values:
    run = Simulation(create(MODEL, x=focus(beta1) + 0.02, y=0.0, beta1=beta1), dt=0.05).run(until=600).trajectory_
    x = run['form.x'][len(run.times) // 2 :]
    peaks = find_peaks(x)[0]
    amplitudes.append((x.max() - x.min()) / 2)
    periods.append(np.mean(np.diff(run.times[peaks])))
fig, (plane, growth) = plt.subplots(1, 2, figsize=(11, 4))
plane.plot(strips[0][:, 0] / 100, -strips[0][:, 1] / 100, label='limit cycle at beta1 = -0.1')
plane.plot(strips[1][:, 0] / 100, -strips[1][:, 1] / 100, label='orbit from near the saddle')
plane.plot(saddle, 0, 'o', color='C3', label='saddle')
plane.set_xlim(-0.9, 1.2)
plane.set_ylim(-0.75, 0.75)
plane.set_xlabel('x')
plane.set_ylabel('y')
plane.set_title('Limit cycle inside the separatrix')
plane.legend(fontsize=8)
growth.plot(values, amplitudes, 'o-', label='amplitude of x')
growth.set_xlabel(r'$\beta_1$')
growth.set_ylabel('amplitude')
growth.axvline(-0.2125, color="black", linestyle=":", label="homoclinic")
twin = growth.twinx()
twin.plot(values, periods, 's-', color='C1', label='period')
twin.set_ylabel('period')
growth.set_title('The cycle grows and slows towards the saddle')
growth.legend(loc='upper center', fontsize=8)
fig.tight_layout()
plt.show()
