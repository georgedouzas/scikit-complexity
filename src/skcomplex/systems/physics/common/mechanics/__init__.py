"""Mechanics realizations shared by the physics packages.

The package holds the swing of a pendulum, the equations of Hamilton and Lagrange, and the Poisson brackets and Noether
charges of a mechanical system.
"""

from ._hamiltonian import HamiltonianInteraction
from ._invariants import NoetherResult, noether, poisson_brackets
from ._lagrangian import LagrangianInteraction
from ._pendulum_swing import PendulumSwingInteraction
from ._symmetry_spec import SymmetrySpec

__all__: list[str] = [
    'HamiltonianInteraction',
    'LagrangianInteraction',
    'NoetherResult',
    'PendulumSwingInteraction',
    'SymmetrySpec',
    'noether',
    'poisson_brackets',
]
