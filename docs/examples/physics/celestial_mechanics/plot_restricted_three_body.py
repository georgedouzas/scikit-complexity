"""
# Restricted three-body problem

A spacecraft moving in the field of the Earth and the Moon, seen in the frame that turns with them, loops around the
Earth on a precessing rosette. It can never enter the region forbidden by its Jacobi constant, the one conserved
quantity of the rotating frame, although at this energy a narrow neck towards the Moon is open.
"""

# %%
# The mass ratio is that of the Earth and the Moon, `μ = 0.01215`, with the Earth at `x = -μ` and the Moon at
# `x = 1 - μ`. The model logs the coordinates as time series only, so the observer below also logs the path of the
# spacecraft and the two primaries in the plane, with `y` negated because the 2D views of Rerun point their vertical
# axis down (remove once the core draws planar positions y-up).

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'restricted_three_body'
MU = 0.01215


# Workaround: remove once the core draws planar trajectories y-up.
class PathObserver(RerunObserver):
    """Observer that also logs the primaries and the path of the spacecraft."""

    def on_start(self, problem):
        super().on_start(problem)
        primaries = rr.Points2D(
            [[-MU, 0.0], [1 - MU, 0.0]],
            radii=[0.06, 0.03],
            colors=[[90, 170, 230], [200, 200, 200]],
        )
        self.recording_.log('frame/primaries', primaries, static=True)
        self.path_ = []

    def on_frame(self, frame):
        self.path_.append([frame.value('body.x'), -frame.value('body.y')])
        super().on_frame(frame)
        if frame.step % self.interval == 0:
            self.recording_.log('frame/path', rr.LineStrips2D([self.path_], radii=0.005, colors=[250, 180, 60]))


blueprint = rrb.Blueprint(
    rrb.Spatial2DView(
        origin='/frame',
        name='rotating frame',
        visual_bounds=rrb.VisualBounds2D(x_range=(-1.0, 1.2), y_range=(-0.9, 0.9)),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, x=0.5, y=0.0, vx=0.0, vy=1.0, mu=MU)
observer = PathObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=8)
simulation = Simulation(system, dt=0.02, observers=[observer]).run(until=40)

# %%
# Jacobi constant and the forbidden region
# ----------------------------------------
# The Jacobi constant `C = x^2 + y^2 + 2 (1 - μ) / r1 + 2 μ / r2 - v^2` should stay constant along the path, up to the
# error of the solver, and the path should never enter the shaded region where `x^2 + y^2 + 2 (1 - μ) / r1 + 2 μ / r2`
# is below `C`, which would need a negative kinetic energy.


def effective(x, y):
    """Return twice the effective potential of the rotating frame, with the sign that makes it positive."""
    return x**2 + y**2 + 2 * (1 - MU) / np.hypot(x + MU, y) + 2 * MU / np.hypot(x - 1 + MU, y)


x, y = np.ravel(simulation.trajectory_['body.x']), np.ravel(simulation.trajectory_['body.y'])
vx, vy = np.ravel(simulation.trajectory_['body.x_d1']), np.ravel(simulation.trajectory_['body.y_d1'])
jacobi = effective(x, y) - vx**2 - vy**2
grid_x, grid_y = np.meshgrid(np.linspace(-1.6, 1.6, 400), np.linspace(-1.6, 1.6, 400))
fig, (plane, drift) = plt.subplots(1, 2, figsize=(10, 4.5))
plane.contourf(grid_x, grid_y, effective(grid_x, grid_y), levels=[0, jacobi[0]], colors=['lightgrey'])
plane.plot(x, y, linewidth=0.8)
plane.plot([-MU, 1 - MU], [0, 0], 'o', color='black')
plane.set_aspect('equal')
plane.set_xlabel('x')
plane.set_ylabel('y')
plane.set_title('Path outside the forbidden region, shaded')
drift.plot(simulation.trajectory_.times, (jacobi - jacobi[0]) / abs(jacobi[0]))
drift.set_xlabel('time')
drift.set_ylabel('relative change of C')
drift.set_title('The Jacobi constant is conserved')
fig.tight_layout()
plt.show()
