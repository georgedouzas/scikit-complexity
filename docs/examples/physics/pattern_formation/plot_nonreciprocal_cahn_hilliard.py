"""
# Nonreciprocal Cahn-Hilliard

A demixing field coupled nonreciprocally to a diffusing one cannot settle into still domains: the domains of one
field chase those of the other, so phase separation turns into traveling bands that stop coarsening. The space-time
image shows the demixing field on a periodic line of length 40, one row per saved step with the earliest at the top,
where the domains draw parallel diagonal stripes: they move along the line at a constant speed instead of
coarsening.
"""

# %%
# The model is created by name from the catalog with 128 sites, the purely nonreciprocal coupling `alpha = 0.3`, and
# the diffusivity `D = 0.1` of the second field, above the threshold `alpha > D` of Brauns and Marchetti. Both fields
# start from noise of amplitude 0.01 around the mixed state. The observer saves every frame of 2 time units with the
# default blueprint, whose primary view is the space-time image of `phi`.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import dispersion_relation, fields_of
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'nonreciprocal_cahn_hilliard'
LENGTH = 40.0

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=2.0, observers=[observer], random_state=0).run(until=400)

# %%
# A propagating unstable band
# ---------------------------
# Around the mixed state, every Fourier mode of wavenumber `q` grows at the rate and turns at the frequency of an
# eigenvalue of the linearised laws. Above threshold the band of unstable modes ends at `q+ = sqrt(1 - D)`, where the
# frequency is `q+^2 sqrt(alpha^2 - D^2)` (Brauns and Marchetti, Eq. 5): the growing modes near the edge propagate.
# Below threshold, at `alpha = 0.05`, the whole band is stationary. The relations are computed on a long line of 512
# sites, whose wavenumbers resolve the band finely. In the run above, the phase of the dominant Fourier
# mode of `phi` should advance at a constant rate once the domains form, the signature of domains that travel.

above = dispersion_relation(create(MODEL, shape=(512,), length=400.0))
below = dispersion_relation(create(MODEL, shape=(512,), length=400.0, alpha=0.05))
edge = np.sqrt(1 - 0.1)
phi = fields_of(simulation.trajectory_, 'mixture.phi')
times = 2.0 * np.arange(len(phi))
modes = np.fft.rfft(phi, axis=1)
mode = 1 + np.argmax(np.abs(modes[-1, 1:]))
phase = np.unwrap(np.angle(modes[:, mode]))
late = times > 150
drift = np.polyfit(times[late], phase[late], 1)[0]
speed = abs(drift) / (2 * np.pi * mode / LENGTH)

fig, (rates, turning) = plt.subplots(1, 2, figsize=(11, 4))
for relation, label in [(above, 'alpha = 0.3'), (below, 'alpha = 0.05')]:
    order = np.argsort(relation.magnitudes)
    rates.plot(relation.magnitudes[order], relation.growth_rates[order], label=f'growth rate, {label}')
    rates.plot(
        relation.magnitudes[order],
        np.abs(relation.frequencies[order]),
        '--',
        color=rates.lines[-1].get_color(),
        label=f'|frequency|, {label}',
    )
rates.axhline(0.0, color='gray', linewidth=0.8)
rates.axvline(edge, color='black', linestyle=':', label='band edge q+ = sqrt(1 - D)')
rates.plot([edge], [edge**2 * np.sqrt(0.3**2 - 0.1**2)], 'k*', markersize=10, label='q+^2 sqrt(alpha^2 - D^2)')
rates.set_xlim(0, 1.2)
rates.set_ylim(-0.3, 0.35)
rates.set_xlabel('wavenumber q')
rates.set_ylabel('growth rate and frequency')
rates.set_title('The unstable band propagates above threshold')
rates.legend(fontsize=7)
turning.plot(times, phase, label=f'mode {mode}')
turning.plot(times[late], np.polyval(np.polyfit(times[late], phase[late], 1), times[late]), 'k--', label='linear fit')
turning.set_xlabel('time')
turning.set_ylabel('unwrapped phase of the dominant mode')
turning.set_title(f'The domains travel at speed {speed:.3f}')
turning.legend()
fig.tight_layout()
plt.show()
