"""
# Lattice Yang-Mills and pure lattice QCD

SU(2) or SU(3) matrices on the links of a periodic lattice sample the Wilson action `-(beta / N) sum_p Re tr U_p`,
where `U_p` is the ordered product of the link matrices around a plaquette. The matrices do not commute, so the order of
the product matters, and SU(3) is the gauge group of quantum chromodynamics. In two dimensions the plaquettes are
independent, so the mean plaquette is a one-plaquette integral over the group. With two time slices, the Polyakov loop
that winds along the time axis orders above a critical coupling, the deconfinement of quarks into a plasma.
"""

# %%
# The plaquette against the one-plaquette integral
# ------------------------------------------------
# On an 8 by 8 lattice the mean plaquette matches `I_2(beta) / I_1(beta)` for SU(2) and the integral over the
# eigenvalue angles of SU(3), weighted by the Haar measure.

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import iv

from skcomplex.systems import create_simulation

SYSTEM = 'lattice_yang_mills'


def _su3_plaquette(beta):
    """Return the mean of `Re tr U / 3` over SU(3) with the weight `exp(beta Re tr U / 3)` and the Haar measure."""
    angles = np.linspace(-np.pi, np.pi, 200, endpoint=False)
    first, second = np.meshgrid(angles, angles)
    phases = np.exp(1j * np.stack([first, second, -first - second]))
    haar = np.prod([np.abs(phases[i] - phases[j]) ** 2 for i, j in ((0, 1), (0, 2), (1, 2))], axis=0)
    traces = phases.sum(axis=0).real / 3
    weights = haar * np.exp(beta * traces)
    return (weights * traces).sum() / weights.sum()


couplings = {2: [0.5, 1.5, 3.0, 5.0], 3: [1.0, 3.0, 6.0, 10.0]}
measured = {}
for colors, betas in couplings.items():
    measured[colors] = [
        create_simulation(SYSTEM, size=8, dimension=2, colors=colors, beta=beta)
        .prepare()
        .run(until=150)
        .trajectory_.values('mean_plaquette')[50:]
        .mean()
        for beta in betas
    ]

grid = np.linspace(0.1, 10.0, 100)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(grid, iv(2, grid) / iv(1, grid), color='tab:blue', label='SU(2), I_2 / I_1')
ax.plot(grid, [_su3_plaquette(beta) for beta in grid], color='tab:orange', label='SU(3), Haar integral')
ax.plot(couplings[2], measured[2], 'o', color='tab:blue', label='SU(2), sampled')
ax.plot(couplings[3], measured[3], 's', color='tab:orange', label='SU(3), sampled')
ax.set_xlabel('beta')
ax.set_ylabel('mean plaquette')
ax.set_title('Two-dimensional Wilson action')
ax.legend()
plt.show()

# %%
# Deconfinement of SU(2)
# ----------------------
# On a 6 by 6 by 6 lattice with two time slices, the modulus of the mean Polyakov loop is small in the confined phase
# and grows above the critical coupling near `beta = 1.9`.

betas = np.linspace(1.4, 2.6, 5)
loops = [
    create_simulation(SYSTEM, size=6, extent=2, beta=beta).prepare().run(until=150).trajectory_.values('polyakov_loop')
    for beta in betas
]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(betas, [series[50:].mean() for series in loops], 'o-')
ax.set_xlabel('beta')
ax.set_ylabel('|mean Polyakov loop|')
ax.set_title('SU(2) with two time slices')
plt.show()

# %%
# Thermalization of pure lattice QCD
# ----------------------------------
# SU(3) links on a 4^4 lattice at `beta = 5`, from random matrices and from the identity, relax to the same mean
# plaquette.

cold = np.broadcast_to(np.eye(3), (1024, 3, 3))
fig, ax = plt.subplots(figsize=(6, 4))
for label, links in (('hot start', None), ('cold start', cold)):
    simulation = create_simulation(SYSTEM, size=4, colors=3, beta=5.0, width=0.3, links=links).prepare()
    ax.plot(simulation.run(until=100).trajectory_.values('mean_plaquette'), label=label)
ax.set_xlabel('sweep')
ax.set_ylabel('mean plaquette')
ax.set_title('SU(3) Wilson action at beta = 5')
ax.legend()
plt.show()
