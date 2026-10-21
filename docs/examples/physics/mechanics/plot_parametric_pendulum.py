"""
# Parametric pendulum

A pendulum whose pivot is shaken up and down at twice its natural frequency swings ever higher from a tiny start, like
a child pumping a swing: parametric resonance pumps energy in at every half swing.
"""

# %%
# The pivot oscillates with relative amplitude `ε = 0.2` at the frequency `Ω = 2 sqrt(g / l)`, and the pendulum starts
# at 0.1 radians at rest.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'parametric_pendulum'


blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/', contents=['+ /swing/theta'], name='angle'),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, theta=0.1, omega=0.0, epsilon=0.2)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=6)
simulation = Simulation(system, dt=0.02, observers=[observer]).run(until=30)

# %%
# Exponential growth at resonance
# -------------------------------
# For small swings the pendulum obeys Mathieu's equation, whose amplitude grows at the rate `ε ω0 / 4` at the principal
# resonance `Ω = 2 ω0`. Started at 0.01 radians, the peaks of `|θ|` should rise parallel to that exponential, dashed,
# below it because a start at rest excites the growing and the decaying solutions equally, until the swings are large
# enough for the nonlinearity to detune them. Without the shaking, `ε = 0`, they should stay at 0.01.

natural = np.sqrt(9.81)
fig, ax = plt.subplots(figsize=(6, 4))
for epsilon in [0.2, 0.0]:
    run = Simulation(create(MODEL, theta=0.01, omega=0.0, epsilon=epsilon), dt=0.01).run(until=40)
    times = np.asarray(run.trajectory_.times)
    angle = np.abs(np.ravel(run.trajectory_['swing.theta']))
    peaks = (angle[1:-1] > angle[:-2]) & (angle[1:-1] >= angle[2:])
    ax.semilogy(times[1:-1][peaks], angle[1:-1][peaks], 'o', markersize=3, label=f'ε = {epsilon}')
ax.semilogy(times, 0.01 * np.exp(0.2 * natural / 4 * times), '--', color='black', label='0.01 exp(ε ω0 t / 4)')
ax.set_ylim(0.005, 4)
ax.set_xlabel('time')
ax.set_ylabel('peak |θ| (radians)')
ax.set_title('Parametric resonance of a shaken pendulum')
ax.legend()
plt.show()
