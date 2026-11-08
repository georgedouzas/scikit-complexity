"""
# Differentiable agents of a COVID-19 epidemic

Agents of a synthetic town live in households of two adults and two children, children go to schools and adults to
companies, and every kind of place spreads the infection with its own transmission. Infection and recovery are relaxed
Bernoulli draws, exact in NumPy runs and smooth in JAX runs, so the run has gradients in the transmissions and
variational inference fits them.
"""

# %%
# The town has 100 households, 4 schools, and 10 companies, with ten agents infected at the start and the
# transmissions 0.9 in households, 0.6 in schools, and 0.3 in companies, as Quera-Bofarull et al. use. Time runs in
# days and the run lasts 30 days. The observer saves every day with the default blueprint.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import calibrate
from skcomplex.systems.biology.epidemiology import DifferentiableCovidAbmSystem

SYSTEM = 'differentiable_covid_abm'
BETAS = {'beta_household': 0.9, 'beta_school': 0.6, 'beta_company': 0.3}

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(DifferentiableCovidAbmSystem(), observers=[observer], random_state=0)

# %%
# Daily infections of children and adults
# ---------------------------------------
# The infections of children and of adults rise as the infection spreads through households, schools, and companies,
# and fall as the town recovers. The counts of a day appear the day after.

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(trajectory.times, trajectory.values('cases.children'), color='tab:orange', label='children')
ax.plot(trajectory.times, trajectory.values('cases.adults'), color='tab:blue', label='adults')
ax.set_xlabel('time (days)')
ax.set_ylabel('daily infections')
ax.set_title('An epidemic in households, schools, and companies')
ax.legend()
plt.show()

# %%
# Variational calibration of the three transmissions
# --------------------------------------------------
# The data are the mean daily infections of children and adults over 20 NumPy runs. Variational inference fits a
# multivariate normal posterior through the JAX run with the Gaussian likelihood, with uniform priors between 0 and 2,
# and averages the loss over fresh draws at every one of its 4000 steps. The posteriors should lie around the true
# transmissions, with the trade-off between households and the other places that Quera-Bofarull et al. find.

times = np.arange(31.0)
runs = [simulate(DifferentiableCovidAbmSystem(), random_state=seed) for seed in range(20)]
paths = ('cases.children', 'cases.adults')
data = {path: (times, np.mean([run.values(path) for run in runs], axis=0)) for path in paths}
posterior = calibrate(
    DifferentiableCovidAbmSystem(),
    data,
    dict.fromkeys(BETAS, (0.0, 2.0)),
    method='variational',
    loss='likelihood',
    n_samples=500,
    budget=4000,
    random_state=0,
)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, (name, truth) in zip(axes, BETAS.items(), strict=True):
    ax.hist(posterior.samples[name], bins=30, color='tab:green', alpha=0.7)
    ax.axvline(truth, color='black', linestyle='--', label='truth')
    ax.set_xlabel(name.replace('_', ' '))
    ax.legend()
axes[0].set_ylabel('posterior draws')
fig.suptitle('Variational posterior of the transmissions')
fig.tight_layout()
plt.show()
