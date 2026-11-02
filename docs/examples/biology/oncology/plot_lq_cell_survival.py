"""
# Linear-quadratic cell survival

The surviving fraction of cells after a dose `D` is `exp(-alpha D - beta G D^2)`, a shoulder on a logarithmic plot,
where the Lea-Catcheside factor `G` holds the repair of sublethal damage. Splitting a dose into fractions spares the
quadratic
term, and 30 fractions of 2 Gy at `alpha / beta = 10` give the biologically effective dose of 72 Gy.
"""

# %%
# The model solves its three equations once, for 30 fractions of 2 Gy, in one frame.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.experiments import Experiment, GridDesign
from skcomplex.models import create
from skcomplex.models.biology.oncology import lq_cell_survival

MODEL = 'lq_cell_survival'

system = create(MODEL, dose=60.0, fractions=30.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer]).check().prepare().run(until=1.0)

# %%
# Survival against the dose
# -------------------------
# An `Experiment` sweeps the total dose of one fraction and of 10 fractions, one static run per value. Fractions turn
# the curve into a nearly straight line of slope `alpha`, so the same dose kills far fewer cells.


def survival(fractions: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the doses and the surviving fractions of a number of fractions."""
    bounds = {'dose': np.linspace(0.0, 20.0, 21).tolist(), 'fractions': [fractions]}
    experiment = Experiment(lq_cell_survival, GridDesign(), bounds, until=1.0, outputs=['cells.survival'])
    result = experiment.check().prepare().run().result_
    return result.column('dose'), result.series('cells.survival')[:, 0]


fig, ax = plt.subplots(figsize=(7, 4))
for fractions in (1.0, 10.0):
    doses, surviving = survival(fractions)
    ax.semilogy(doses, surviving, 'o-', label=f'{fractions:.0f} fractions')
ax.set_xlabel('total dose D (Gy)')
ax.set_ylabel('surviving fraction')
ax.set_title('Fractionation spares the quadratic term')
ax.legend()
fig.tight_layout()
plt.show()
