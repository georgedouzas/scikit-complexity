"""
# Carbon-13 metabolic flux analysis

Labelled atoms of a tracer spread through a metabolic network along its atom transitions, so the labelling of a product
carries the fluxes that made it. Fitting the fluxes to measured labelling recovers fluxes that no balance fixes, here
the split of 2-oxoglutarate in the TCA example of Antoniewicz et al. (2007).
"""

# %%
# The system is built from its class, and the data come from its known fluxes under two parallel tracers with noise of
# standard deviation 0.002 at seed 0. The network with the estimated fluxes runs once, and the primary view is its
# graph.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.systems_biology import Carbon13MFASystem, tca_labelling_analysis

SYSTEM = 'carbon13_mfa'
TRUE = {'v1': 100.0, 'v2': 100.0, 'v3': 50.0, 'v4': 50.0, 'v5': 50.0, 'v6': 50.0, 'v8': 50.0, 'v9': 50.0}

system = Carbon13MFASystem().prepare().system_
labelling = tca_labelling_analysis(noise=0.002, random_state=0)
result = labelling(system)
network = system.elements[0]
fixed = {f'{bound}_{name}': result.fluxes[name] for name in TRUE for bound in ('lower', 'upper')}
estimated = CompositeSystem(elements=[type(network)(network.label).set_params(**fixed)], time=StaticTime())
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='tca/graph')
simulate(estimated, until=0.0, observers=[observer])

# %%
# Estimated fluxes against the true ones
# --------------------------------------
# The fit should recover every net flux, with a 95% interval that holds its true value, and the chi-square test should
# accept it at the 5% level.

names = list(TRUE)
values = np.array([result.fluxes[name] for name in names])
lower = np.array([result.intervals[name][0] for name in names])
upper = np.array([result.intervals[name][1] for name in names])
truth = np.array([TRUE[name] for name in names])
fig, ax = plt.subplots(figsize=(9, 4))
positions = np.arange(len(names))
ax.errorbar(
    positions,
    values,
    yerr=[values - lower, upper - values],
    fmt='o',
    capsize=4,
    label='estimate, 95% interval',
)
ax.plot(positions, truth, 'x', color='tab:red', label='true flux')
ax.set_xticks(positions, names)
ax.set_ylabel('net flux')
ax.legend()
verdict = 'accepts' if result.accepted else 'rejects'
ax.set_title(f'v3 {result.fluxes["v3"]:.1f} and v4 {result.fluxes["v4"]:.1f}; the test {verdict} the fit')
fig.tight_layout()
plt.show()

# %%
# Measured and simulated glutamate
# --------------------------------
# The simulated distributions of glutamate at the estimate should lie on the measured ones in both experiments.

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
measured = {item.experiment: item.distribution for item in labelling.keywords['measurements']}
for ax, (experiment, labels) in zip(axes, result.distributions.items(), strict=True):
    atoms = np.arange(labels['Glu'].size)
    ax.bar(atoms - 0.2, measured[experiment], width=0.4, label='measured')
    ax.bar(atoms + 0.2, labels['Glu'], width=0.4, label='simulated')
    ax.set_xlabel('labelled atoms')
    ax.set_title(f'experiment {experiment}')
axes[0].set_ylabel('fraction of glutamate')
axes[0].legend()
gap = max(np.max(np.abs(measured[name] - labels['Glu'])) for name, labels in result.distributions.items())
fig.suptitle(f'The largest gap between measured and simulated fractions is {gap:.4f}')
fig.tight_layout()
plt.show()
