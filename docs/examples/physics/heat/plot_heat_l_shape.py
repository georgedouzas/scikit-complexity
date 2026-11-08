"""
# Heat on an L-shaped plate

The cold plate warms from its boundary, where the temperature is fixed, until it reaches the steady state of Laplace's
equation: a smooth harmonic surface drawn with the temperature as its height and colour.
"""

# %%
# The temperature lives on the nodes of a finite element mesh. The 3D view looks at the surface from above one corner.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.heat import HeatLShapeSystem

SYSTEM = 'heat_l_shape'

camera = rrb.EyeControls3D(position=(2.6, -2.2, 1.4), look_target=(0.0, 0.0, 0.2), eye_up=(0.0, 0.0, 1.0))
blueprint = rrb.Blueprint(
    rrb.Spatial3DView(origin='/plate/u', name='u', eye_controls=camera, line_grid=False),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
system = HeatLShapeSystem()
trajectory = simulate(system, observers=[observer])

# %%
# Approach to the steady state
# ----------------------------
# The boundary temperature is the harmonic function `exp(x) sin(y)`, so the steady state is that function on the
# whole plate. The largest difference between the simulated temperature and it should decay exponentially until it
# reaches the discretization error of the mesh, and at the last frame the temperature at every node should lie on the
# line of the exact solution.

nodes = system.prepare().system_.elements[0].space.prepare().positions_
exact = HeatLShapeSystem().temperature(nodes[:, 0], nodes[:, 1])
temperature = trajectory.values('plate.u')
fig, (agreement, decay) = plt.subplots(1, 2, figsize=(10, 4))
agreement.plot(exact, temperature[-1], '.', markersize=4)
agreement.plot([exact.min(), exact.max()], [exact.min(), exact.max()], color='black', linewidth=1)
agreement.set_xlabel('exact steady temperature')
agreement.set_ylabel('simulated temperature at the end')
agreement.set_title('Nodes lie on the exact solution')
decay.semilogy(trajectory.times, np.abs(temperature - exact).max(axis=1))
decay.set_xlabel('time')
decay.set_ylabel('largest error')
decay.set_title('Exponential approach to the steady state')
fig.tight_layout()
plt.show()
