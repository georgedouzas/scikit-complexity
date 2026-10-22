"""
# Active Brownian particles with local alignment

The swimmers start at random headings. The alignment torque between neighbours spreads one heading through the box, so
the swimmers end up moving together, and the points take one colour, which shows the heading.
"""

# %%
# The swimmers move in a periodic box of side 20 at the density 0.5, and the alignment strength is eight times its
# threshold. The blueprint shows the whole box and draws the swimmers larger than the model's visual does, so they
# read in the gallery thumbnail.

import math

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'aligning_active_brownian_particles'

box = rrb.Spatial2DView(
    origin='/swimmers',
    name='swimmers',
    # Points are drawn with y negated so that y points up, so the vertical range is mirrored.
    visual_bounds=rrb.VisualBounds2D(x_range=(0.0, 20.0), y_range=(-20.0, 0.0)),
    overrides={'swimmers': rr.Points2D.from_fields(radii=0.3).visualizer()},
)
blueprint = rrb.Blueprint(
    box,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, gamma=1.0, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(
    system,
    dt=1.0,
    solver=EulerMaruyama(dt=0.05),
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(
    until=100,
)

# %%
# The alignment transition
# ------------------------
# The mean-field Fokker-Planck equation predicts that the disordered state is unstable when the alignment strength
# exceeds `gamma_c = 2 D_r / (pi d0^2 rho)`, which is about 0.13 here. The stationary polarization of short runs at
# several strengths should stay at the finite-size level, near `1 / sqrt(N)`, below the threshold, and rise towards
# one above it.

D_r, d0, density = 0.1, 1.0, 200 / 20.0**2
threshold = 2 * D_r / (math.pi * d0**2 * density)
strengths = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0])
polarizations = []
for gamma in strengths:
    model_system = create(MODEL, gamma=gamma, random_state=1)
    run = Simulation(
        model_system,
        dt=1.0,
        solver=EulerMaruyama(dt=0.05),
        random_state=1,
        observables=describe(MODEL).observables_of(model_system),
    )
    polarizations.append(np.abs(run.run(until=150).trajectory_['polarization'][75:]).mean())
fig, (series, transition) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(simulation.trajectory_.times, np.abs(simulation.trajectory_['polarization']))
series.set_ylim(0, 1)
series.set_xlabel('time')
series.set_ylabel('polarization')
series.set_title(f'Ordering at gamma = 1.0 = {1 / threshold:.1f} gamma_c')
transition.plot(strengths / threshold, polarizations, 'o-', label='simulation')
transition.axvline(1.0, color='black', linestyle='--', label='mean-field threshold')
transition.axhline(1 / math.sqrt(200), color='grey', linestyle=':', label='1 / sqrt(N)')
transition.set_xscale('log')
transition.set_xticks([0.2, 0.5, 1.0, 2.0, 5.0], labels=['0.2', '0.5', '1', '2', '5'], minor=False)
transition.set_xticks([], minor=True)
transition.set_ylim(0, 1)
transition.set_xlabel('gamma / gamma_c')
transition.set_ylabel('stationary polarization')
transition.set_title('Polarization against the alignment strength')
transition.legend()
fig.tight_layout()
plt.show()
