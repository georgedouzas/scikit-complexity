"""
# Heterogeneous mean-field SIR

Above the epidemic threshold of a scale-free network, the infected densities of all degree classes rise and fall in
one outbreak while the susceptible densities fall to a plateau and the recovered densities rise to the final epidemic
size, the hubs fastest.
"""

# %%
# The model is created by name from the catalog on degrees from 3 to 30 with `P(k) ~ k^-2.5`, at a transmission
# rate per contact of 2.5 times the threshold `<k> / <k^2>` of the unit recovery rate. The observer saves every frame
# of the outbreak with the default blueprint, whose primary view holds the densities of the 28 degree classes.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'heterogeneous_mean_field_sir'

degrees = np.arange(3, 31, dtype=float)
probabilities = degrees**-2.5 / np.sum(degrees**-2.5)
first, second = np.sum(degrees * probabilities), np.sum(degrees**2 * probabilities)
system = create(MODEL, k_max=30, beta=2.5 * first / second)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer], observables=describe(MODEL).observables_of(system)).run(
    until=40,
)

# %%
# Final size against the transmission rate
# ----------------------------------------
# Linearizing the equations of the degree classes, an outbreak grows when `beta / mu > <k> / (<k^2> - <k>)`, close to
# `<k> / <k^2>` on this heterogeneous network. The final epidemic size of runs over a range of transmission rates
# should stay at the initial infected share below the threshold and grow continuously above it. In the outbreak shown
# above, the infected density of a degree class should peak earlier the larger its degree.

rates = np.linspace(0.2, 3.0, 15) * first / second
final_sizes = [
    Simulation(
        create(MODEL, k_max=30, beta=rate, infected=1e-6),
        dt=1.0,
        observables=describe(MODEL).observables_of(system),
    )
    .run(until=300)
    .trajectory_['recovered'][-1]
    for rate in rates
]
infected = simulation.trajectory_['network.i']

fig, (sizes, timing) = plt.subplots(1, 2, figsize=(10, 4))
sizes.plot(rates, final_sizes, 'o-', label='simulated')
sizes.axvline(first / second, color='grey', linestyle='--', label='<k> / <k^2>')
sizes.axvline(first / (second - first), color='black', linestyle=':', label='<k> / (<k^2> - <k>)')
sizes.set_xlabel('transmission rate beta (mu = 1)')
sizes.set_ylabel('final epidemic size')
sizes.set_title('An outbreak above the threshold')
sizes.legend()
for degree in (3, 6, 12, 30):
    density = infected[:, degree - 3]
    timing.plot(simulation.trajectory_.times, density / density.max(), label=f'k = {degree}')
timing.set_xlim(0, 20)
timing.set_xlabel('time')
timing.set_ylabel('infected density / its peak')
timing.set_title('Hubs are infected first')
timing.legend()
fig.tight_layout()
plt.show()
