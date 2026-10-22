"""
# Mattis spin glass

Below the Ising critical temperature the overlap `m_xi = sum_i xi_i s_i / N` of the spins with the hidden random
pattern grows from zero to the Onsager magnetization, while the magnetization `m = sum_i s_i / N` stays near zero. The
time series show both observables during the run.
"""

# %%
# The model is created by name from the catalog on a 48 x 48 lattice at `T = 1.5 J`, starting from random spins. One
# unit of time is one sweep of Metropolis updates. The observer saves every frame with the default blueprint, whose
# primary view is the time series of the overlap and the magnetization.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'mattis_spin_glass'

system = create(MODEL, size=48, temperature=1.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    dt=2,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=400)

# %%
# Hidden order and the gauge transformation
# -----------------------------------------
# The gauge variables `sigma_i = xi_i s_i` follow the Ising model, so the overlap should match Onsager's spontaneous
# magnetization `(1 - sinh(2 J / T)^-4)^(1/8)` below `T_c = 2.269 J` and vanish above it, while the magnetization stays
# near zero at every temperature. The left panel shows the spins and the gauge variables at the end of the run above:
# the spins look disordered and the gauge variables form one domain.

lattice = simulation.trajectory_.lattice('spins.spin')[-1]
pattern = system.elements[0].values['pattern'].reshape(lattice.shape)
temperatures = np.linspace(1.0, 3.5, 11)
overlaps, magnetizations = [], []
for temperature in temperatures:
    model_system = create(MODEL, size=32, temperature=temperature)
    trajectory = Simulation(
        model_system,
        dt=10,
        random_state=0,
        observables=describe(MODEL).observables_of(model_system),
    ).run(until=1000)
    overlaps.append(np.abs(trajectory.trajectory_['overlap'][-30:]).mean())
    magnetizations.append(np.abs(trajectory.trajectory_['magnetization'][-30:]).mean())
critical = 2 / np.log(1 + np.sqrt(2))
grid = np.linspace(1.0, critical - 1e-6, 100)
fig, (spins, gauge, curve) = plt.subplots(1, 3, figsize=(13, 4))
spins.imshow(lattice, cmap='gray')
spins.set_title('Spins s_i')
spins.set_xlabel('column')
spins.set_ylabel('row')
gauge.imshow(pattern * lattice, cmap='gray')
gauge.set_title('Gauge variables xi_i s_i')
gauge.set_xlabel('column')
gauge.set_ylabel('row')
curve.plot(grid, (1 - np.sinh(2 / grid) ** -4) ** (1 / 8), color='black', label='Onsager')
curve.plot(temperatures, overlaps, 'o', label='overlap |m_xi|')
curve.plot(temperatures, magnetizations, 's', label='magnetization |m|')
curve.set_xlabel('temperature T / J')
curve.set_ylabel('order parameter')
curve.set_title('The overlap orders like the Ising model')
curve.legend()
fig.tight_layout()
plt.show()
