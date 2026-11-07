"""
# Delayed Kuramoto

Phase oscillators that hear each other only after a delay still lock to one common rhythm, but not at their mean
natural frequency: the delay shifts the collective frequency, and several locked states can coexist.
"""

# %%
# Five oscillators with frequencies spread over `[0.9, 1.1]` are coupled all to all with `K = 2` through signals delayed
# by `τ = 1`. The default blueprint shows the unwrapped phases, which settle on parallel lines of one slope.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.synchronization import DelayedKuramotoSystem

SYSTEM = 'delayed_kuramoto'

system = DelayedKuramotoSystem(n_oscillators=5, K=2.0, tau=1.0, spread=0.2, seed=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(system, observers=[observer])

# %%
# Collective frequency against the delay
# --------------------------------------
# A nearly in-phase locked state turns at a frequency `Ω` that solves `Ω = 1 - K' sin(Ω τ)`, with the effective
# coupling `K' = K (N - 1) / N` since no oscillator hears itself, and it is stable where `cos(Ω τ) > 0` (Yeung and
# Strogatz), drawn in black. Up to `τ ≈ 2` the frequencies measured over a sweep of the delay should lie on the
# lowest branch. Beyond it the runs, started from phases spread around the circle, settle in locked states that are
# not in phase, which the formula does not describe, so the measured frequencies leave the branches.

effective = 2.0 * 4 / 5
delays = np.linspace(0.1, 3.0, 30)
measured = []
for delay in delays:
    run = simulate(DelayedKuramotoSystem(n_oscillators=5, K=2.0, tau=delay, spread=0.2, seed=0), until=150)
    times, phase = np.asarray(run.times), np.ravel(run.values('oscillator_0.theta'))
    late = times > 100
    measured.append(np.polyfit(times[late], phase[late], 1)[0])
omegas = np.linspace(-1.0, 3.0, 4000)
grid_delays = np.linspace(0.05, 3.0, 600)
fig, ax = plt.subplots(figsize=(6, 4))
for delay in grid_delays:
    residual = omegas - 1 + effective * np.sin(omegas * delay)
    roots = omegas[1:][np.sign(residual[1:]) != np.sign(residual[:-1])]
    stable = roots[np.cos(roots * delay) > 0]
    ax.plot(np.full(stable.size, delay), stable, '.', color='black', markersize=1)
ax.plot(delays, measured, 'o', color='tab:orange', label='simulated')
ax.set_xlabel('delay τ')
ax.set_ylabel('collective frequency Ω')
ax.set_title("Stable solutions of Ω = 1 - K' sin(Ω τ)")
ax.legend()
plt.show()
