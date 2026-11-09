"""
# Random Lotka-Volterra community with higher-order interactions

Every species of a community feels the combined effect of every pair of other species, through coefficients
drawn at random with a competitive mean (Sidhom and Galla 2024). The abundances settle at a unique fixed point where
some species have gone extinct, and the distribution of the survivors is a clipped Gaussian that the dynamic mean-field
theory of the paper predicts.
"""

# %%
# The system is built from its class, with the parameters of the paper for third-order interactions, `mu = -4`, `sigma
# = 1`, and uncorrelated coefficients. It has 100 species here to keep the page quick, half the 200 of the paper, so
# its tensor holds a million entries, and an adaptive solver takes the long steps the slow approach allows. The
# primary view shows the abundances.

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import HigherOrderRandomLotkaVolterraSystem

SYSTEM = 'higher_order_random_lotka_volterra'
MU, SIGMA, EXTINCT = -4.0, 1.0, 1e-6

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
solver = ScipyODESolver('RK45', rtol=1e-6, atol=1e-9)
trajectory = simulate(HigherOrderRandomLotkaVolterraSystem(size=100), solver=solver, observers=[observer])

# %%
# Survivors against the theory
# ----------------------------
# At the fixed point a species has the abundance `x* = max(1 + mu M^2 + z sigma_s, 0)` for a standard Gaussian `z`,
# with `sigma_s^2 = 3 sigma^2 q^2 / 2` (Eqs. 11 and 14). The species with `z` below `z1 = -(1 + mu M^2) / sigma_s` go
# extinct, so `phi = I_0(z1)` survive. The theory gives `M` and `q` from `I_1(z1)` and `I_2(z1)` (Eqs. 17, 18, and 23).
# The histogram of the survivors should follow the Gaussian, and the mean abundance should approach `M`.


def integral(order, z1):
    """Return the integral of `(z - z1)^order` over the standard Gaussian above `z1`, Eq. 18."""
    return quad(lambda z: (z - z1) ** order * norm.pdf(z), z1, np.inf)[0]


def mean_abundance(z1):
    """Return the mean abundance of the threshold `z1`, the root of Eq. 23 that the dynamics reach."""
    linear = z1 / integral(1, z1)
    return (-linear - np.sqrt(linear**2 - 4 * MU)) / (2 * MU)


def heterogeneity(z1):
    """Return the `sigma` of the threshold `z1` with uncorrelated coefficients, Eq. 32."""
    return np.sqrt(2 / 3) * integral(1, z1) / (mean_abundance(z1) * integral(2, z1))


z1 = brentq(lambda z: heterogeneity(z) - SIGMA, -6.0, -0.84)
M = mean_abundance(z1)
q = M**2 * integral(2, z1) / integral(1, z1) ** 2
sigma_s = np.sqrt(1.5) * SIGMA * q
abundances = trajectory.get_values('species.x')
final = abundances[-1]
fig, (mean, histogram) = plt.subplots(1, 2, figsize=(10, 4))
mean.plot(trajectory.times, abundances.mean(axis=1), label='mean abundance')
mean.plot(trajectory.times, (abundances > EXTINCT).mean(axis=1), label='surviving fraction')
mean.axhline(M, color='C0', linestyle='--')
mean.axhline(integral(0, z1), color='C1', linestyle='--')
mean.set_xlabel('time')
mean.set_title('The community settles at the predicted values')
mean.legend()
values = np.linspace(0.0, final.max(), 200)
histogram.hist(final[final > EXTINCT], bins=20, density=True, alpha=0.6, label='survivors')
histogram.plot(values, norm.pdf(values, 1 + MU * M**2, sigma_s) / integral(0, z1), color='k', label='theory, Eq. 14')
histogram.set_xlabel('abundance at the fixed point')
histogram.set_ylabel('density')
histogram.set_title(f'{np.mean(final > EXTINCT):.1%} survive, the theory gives {integral(0, z1):.1%}')
histogram.legend()
fig.tight_layout()
plt.show()
