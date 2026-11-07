"""
# Inverse iteration of Julia sets

The quadratic map `z -> z² + omega` pushes points away from its Julia set, so its two inverse branches `±√(z - omega)`
pull them onto it. Choosing a branch at random at every step, a cloud of points settles on the Julia set within a few
steps: a random function system with a fractal attractor.
"""

# %%
# Five thousand points iterate 30 steps at Douady's rabbit, `omega = -0.123 + 0.745 i`. After the run the points of
# the last step are logged as points of the plane, which the blueprint shows.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'inverse_julia'

blueprint = rrb.Blueprint(
    rrb.Spatial2DView(
        origin='/julia',
        name='Julia set',
        visual_bounds=rrb.VisualBounds2D(x_range=[-150, 150], y_range=[-120, 120]),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=30)
simulation = Simulation(create_system(SYSTEM), dt=1, observers=[observer]).prepare().run(until=30)
z = simulation.trajectory_.values('points.z')[-1]
observer.recording_.log(
    'julia',
    rr.Points2D(100 * np.column_stack([z.real, -z.imag]), radii=0.4, colors=[90, 170, 255]),
    static=True,
)
observer.recording_.flush()

# %%
# Julia sets and the weight of the branches
# -----------------------------------------
# The attractor should be the unit circle for `omega = 0`, connected sets such as the basilica and Douady's rabbit for
# constants inside the Mandelbrot set, and dust for a constant outside it, such as `omega = 0.3`. Weighting the
# positive root more, `p = 0.9`, should crowd the points on part of the set without changing the set (Dilão, Exercise
# 5.8).

cases = [(0.0, 0.5), (-1.0, 0.5), (-0.123 + 0.745j, 0.5), (-0.123 + 0.745j, 0.9), (0.3, 0.5), (-0.8 + 0.2j, 0.5)]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for axis, (omega, p) in zip(axes.ravel(), cases, strict=True):
    run = Simulation(create_system(SYSTEM, omega=omega, p=p, n_points=8000), dt=1).prepare().run(until=40)
    points = run.trajectory_.values('points.z')[-1]
    axis.plot(points.real, points.imag, ',', color='C0')
    axis.set_aspect('equal')
    axis.set_title(f'omega = {omega:g}, p = {p}')
fig.tight_layout()
plt.show()
