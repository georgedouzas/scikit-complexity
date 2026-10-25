"""Backends that run compiled problems: NumPy, Numba, and LAMMPS for particles.

Importing the package registers every backend and the solvers that run in them.
"""

from ._base import ExternalSolver
from ._numba import NumbaBackend
from ._numpy import NumpyBackend

__all__: list[str] = [
    'ExternalSolver',
    'NumbaBackend',
    'NumpyBackend',
]
