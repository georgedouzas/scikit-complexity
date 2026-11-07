"""
# Eisenberg-Noe clearing

Banks owe each other payments, and a bank that cannot pay in full pays what it has, which leaves its creditors short
in turn. The clearing vector of Eisenberg and Noe is the set of payments consistent with all these shortfalls, the
fixed point of proportional payments that the system reaches round by round.
"""

# %%
# Fifty banks with random liabilities and exponential external assets start by promising full payments. The default
# blueprint shows the total shortfall and the number of defaulting banks round by round.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.finance import EisenbergNoeClearingSystem

SYSTEM = 'eisenberg_noe_clearing'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(EisenbergNoeClearingSystem(), observers=[observer])

# %%
# The fixed point as a linear programme
# -------------------------------------
# The greatest clearing vector maximizes the total payment subject to `p <= p_bar` and `p <= e + Pi^T p`. The payments
# the iteration reaches should lie on the diagonal against the solution of that programme by SciPy.

system = EisenbergNoeClearingSystem()
parts = system.prepare().system_
matrix = np.asarray(parts.interactions[0].liabilities)
external = parts.elements[0].values['external']
owed = matrix.sum(axis=1)
shares = np.divide(matrix, owed[:, None], out=np.zeros_like(matrix), where=owed[:, None] > 0)
size = len(owed)
solution = linprog(
    -np.ones(size),
    A_ub=np.eye(size) - shares.T,
    b_ub=external,
    bounds=list(zip(np.zeros(size), owed, strict=True)),
)
payments = trajectory.agents('banks', len(trajectory.times) - 1).values['payment']
fig, (rounds, check) = plt.subplots(1, 2, figsize=(10, 4))
rounds.plot(trajectory.times[1:], trajectory.values('clearing.shortfall')[1:])
rounds.set_xlabel('round')
rounds.set_ylabel('total shortfall')
rounds.set_title('The iteration from full payments')
check.plot(solution.x, payments, 'o')
check.plot([0, owed.max()], [0, owed.max()], 'k:', lw=0.8)
check.set_xlabel('payment of the linear programme')
check.set_ylabel('payment of the iteration')
check.set_title('The iteration reaches the clearing vector')
fig.tight_layout()
plt.show()
