"""Mechanics realizations shared by the physics packages.

The package holds pendulums, reflections off walls, Hamilton's equations, and the Poisson brackets and Noether charges
of a mechanical system.
"""

from ._hamiltonian import HamiltonianInteraction
from ._invariants import NoetherResult, noether, poisson_brackets
from ._pendulum import PendulumElement
from ._pendulum_interaction import PendulumInteraction
from ._reflection import ReflectionInteraction
from ._symmetry_spec import SymmetrySpec

__all__: list[str] = [
    'HamiltonianInteraction',
    'NoetherResult',
    'PendulumElement',
    'PendulumInteraction',
    'ReflectionInteraction',
    'SymmetrySpec',
    'noether',
    'poisson_brackets',
]
