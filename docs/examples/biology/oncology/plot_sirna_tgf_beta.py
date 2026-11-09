"""
# siRNA against TGF-beta

The tumour secretes TGF-beta, which blinds the immune system, and escapes to its carrying capacity of nearly two
billion cells within weeks.
"""

# %%
# The system is built from its class with the book's parameters, a thousand effector and tumour cells, and no siRNA,
# and runs for 200 days. The observer saves every frame with the default blueprint, whose primary view is the time
# series of the cells, the cytokines, and the siRNA.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import SirnaTGFBetaSystem

SYSTEM = 'sirna_tgf_beta'

system = SirnaTGFBetaSystem().prepare().system_
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, dt=1.0, observers=[observer], until=200)

# %%
# Immune escape and its reversal
# ------------------------------
# Without TGF-beta the effector cells hold the tumour near ten thousand cells. With TGF-beta it escapes, and siRNA
# infused at growing rates lowers the TGF-beta it makes. The tumour should fall as the dose grows, and the largest dose
# should clear it.

solver = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-10)
arms = {
    'no TGF-beta': SirnaTGFBetaSystem(rho4=0.0).prepare().system_,
    'TGF-beta, no siRNA': system,
    'siRNA 1e6 per day': SirnaTGFBetaSystem(u=1e6).prepare().system_,
    'siRNA 1e8 per day': SirnaTGFBetaSystem(u=1e8).prepare().system_,
}
fig, (tumour, cytokine) = plt.subplots(1, 2, figsize=(10, 4))
for name, arm in arms.items():
    trajectory = simulate(arm, dt=1.0, solver=solver, until=200)
    tumour.semilogy(trajectory.times, trajectory.get_values('site.A').clip(1.0), label=name)
    cytokine.semilogy(trajectory.times, trajectory.get_values('site.C2').clip(1e-3), label=name)
tumour.set_xlabel('time (days)')
tumour.set_ylabel('tumour cells')
tumour.set_title('TGF-beta lets the tumour escape')
tumour.legend()
cytokine.set_xlabel('time (days)')
cytokine.set_ylabel('TGF-beta (pg/ml)')
cytokine.set_title('siRNA lowers TGF-beta')
fig.tight_layout()
plt.show()
