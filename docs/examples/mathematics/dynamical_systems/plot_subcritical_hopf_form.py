"""
# Subcritical Hopf form

The parameter `alpha` returns to -0.1 three times, and the system answers with large oscillations, with none, and with
large oscillations again. At this value the focus and the limit cycle are both stable, and the system stays on the
attractor it had before, a hysteresis loop.
"""

# %%
# The run starts on the limit cycle at `alpha = -0.1`. The schedule then lowers `alpha` to -0.35, below the fold of the
# cycle at `-1 / (4 gamma) = -0.25`, so the oscillation dies. Back at -0.1 the focus holds. Raising `alpha` to 0.3
# destabilizes the focus and the oscillation grows back, and back at -0.1 it stays. The default blueprint shows `x`
# and `y` as time series.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'subcritical_hopf_form'
SWEEP = [(0.0, -0.1), (50.0, -0.35), (100.0, -0.1), (150.0, 0.3), (230.0, -0.1)]


def alpha_at(time):
    """Return the value of alpha that the sweep holds at a time."""
    return [value for start, value in SWEEP if start <= time][-1]


system = create(MODEL, x=1.0, alpha=-0.1)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, observers=[observer], schedule={'form.alpha': alpha_at}).run(until=290)

# %%
# Amplitude along the sweep and the two attractors
# ------------------------------------------------
# The amplitude `r = |z|` of the run should sit on the cycle radius `r² = (1 + √(1 + 4 alpha)) / 2` during the first and
# the last stays at `alpha = -0.1`, and near 0 during the middle one. Started from a small and from a large amplitude,
# runs at fixed `alpha` should end on different attractors exactly in the bistable window `-1/4 < alpha < 0`.

trajectory = simulation.trajectory_
amplitude = np.hypot(trajectory['form.x'], trajectory['form.y'])
alphas = np.linspace(-0.4, 0.2, 25)
finals = {
    start: [
        np.hypot(*Simulation(create(MODEL, x=start, alpha=alpha), dt=1.0).run(until=150).state_) for alpha in alphas
    ]
    for start in (0.05, 1.2)
}
branch = np.linspace(-0.25, 0.2, 200)
fig, (sweep, attractors) = plt.subplots(1, 2, figsize=(11, 4))
sweep.plot(trajectory.times, amplitude, label='amplitude $r$')
sweep.plot(trajectory.times, [alpha_at(time) for time in trajectory.times], label=r'$\alpha$')
sweep.axhline(np.sqrt((1 + np.sqrt(1 - 0.4)) / 2), color='black', linestyle=':', label=r'cycle at $\alpha = -0.1$')
sweep.set_xlabel('time')
sweep.set_ylabel('value')
sweep.set_title('Same alpha, different attractors')
sweep.legend(fontsize=8)
attractors.plot(branch, np.sqrt((1 + np.sqrt(1 + 4 * branch)) / 2), color='black', label='stable cycle')
attractors.plot(branch[branch < 0], np.sqrt((1 - np.sqrt(1 + 4 * branch[branch < 0])) / 2), '--', color='black')
attractors.plot([-0.4, 0.0], [0.0, 0.0], color='black')
attractors.plot(alphas, finals[0.05], 'o', fillstyle='none', label='from $r = 0.05$')
attractors.plot(alphas, finals[1.2], 'x', label='from $r = 1.2$')
attractors.axvspan(-0.25, 0.0, alpha=0.15, label='bistable window')
attractors.set_xlabel(r'$\alpha$')
attractors.set_ylabel('final amplitude')
attractors.set_title('Focus, stable and unstable (dashed) cycles')
attractors.legend(fontsize=8)
fig.tight_layout()
plt.show()
